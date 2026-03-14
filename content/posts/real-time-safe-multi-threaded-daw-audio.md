+++
title = "Real-Time Safe Multi-Threaded DAW Audio"
date = 2026-03-09

[extra]
math = true
mermaid = true
+++

# The problem statement

At a high level, audio processing is done in chunks.

- The driver gives you a chunk of input data to process; you process it and return a chunk of output data. This is called the *audio callback*.
- While that output data is being played back, the driver gives you the next chunk of input data to process.
- If the next chunk of output data isn't ready by the time the previous chunk is done playing, the audio output glitches for a bit. This is called a *buffer underrun*.

Ideally, buffer underruns never happen, so the audio callback must always finish before the next buffer is needed. For example, at a sample rate of 44.1 kHz and a buffer size of 512 samples, the callback must always finish in under about 11.6 ms.

Additionally, the audio callback typically runs in a high-priority thread, which must never block. A good rule of thumb is to avoid the following operations:

- acquiring a lock, which may cause [priority inversion](https://en.wikipedia.org/wiki/Priority_inversion) [^1]
- interacting with a heap allocator, which may acquire allocator-internal or OS-internal locks
- doing any I/O, which may acquire OS-internal locks or interact with a heap allocator
- calling any code with unpredictable or poor worst-case runtime
- calling any code that may violate the prior requirements
	- this includes most system calls

Code that abides by these requirements is considered "real-time safe" (not to be confused with "memory safe").

[^1]: [Priority inheritance](https://en.wikipedia.org/wiki/Priority_inheritance) is one way around this, but waiting for a lock may still have unpredictable runtime.

---

# The audio graph

One of the main components of a DAW engine is an audio graph, which abstractly represents the work to be done between receiving input audio and emitting output audio as a [directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) (DAG) of processing nodes (e.g., playlist tracks, mixer channels) and the dependencies between them. A node may only be processed after all of its dependencies have been processed, and every node must be visited in every audio callback.

{% mermaid() %}
stateDiagram-v2
[*] --> A
  A --> D
  A --> B
  A --> C
  B --> D
  C --> D
  C --> E
  C --> F
  D --> F
  E --> F
  F --> [*]
{% end %}

---

# Single-threading

In a single-threaded environment, processing typically happens as follows:

Given a DAG, we can interpret the edges as a partial order over the nodes, where an edge $A\rightarrow B$ means $A$ must be processed before $B$. We can then sort the nodes in ascending order based on this partial order, which results in a list of nodes in such order that every node's dependencies are ordered before it itself is. This is called a [topological sort](https://en.wikipedia.org/wiki/Topological_sorting) of the DAG. Note that the topological sort of a DAG isn't necessarily unique, since the partial order produced usually isn't also a total order.

## Depth-first search

Nodes can be sorted using dependency comparisons, but a more efficient method is DFS-based topological sort, which runs in linear time ($\mathcal{O}(|V|+|E|)$). By pushing a node after all of its dependents, we get a reverse topological sort. Since the audio graph is known to be acyclic, we can omit explicit cycle detection.

```rs
fn create_schedule(nodes: &mut [Node], schedule: &mut Vec<usize>) {
    fn visit(curr: usize, nodes: &mut [Node], schedule: &mut Vec<usize>) {
        if nodes[curr].visited {
            return;
        }

        for next in &nodes[curr].outgoing {
            visit(next, nodes, schedule);
        }

        schedule.push(curr);

        nodes[curr].visited = true;
    }

    for node in 0..nodes.len() {
        visit(node, nodes, schedule);
    }

    schedule.reverse();
}
```

Afterwards, we can simply iterate through `schedule`, processing one node at a time:

```rs
fn run_schedule(nodes: &mut [Node], schedule: &[usize]) {
    for node in schedule {
        nodes[node].sum_inputs(nodes);
        nodes[node].process();
    }
}
```

Quite trivially, we can see that the runtime of the audio processing is bounded by the amount of time every node in the graph spends processing. That's quite predictable, but large projects can easily exceed the work that fits in the callback time budget. For example, this is one potential execution order of the above audio graph example:

{% mermaid() %}
gantt
  dateFormat SSS
  axisFormat %L
  deadline: milestone, 050,
  A:        a,         000,     4ms
  B:        b,         after a, 27ms
  C:        c,         after b, 8ms
  D:        d,         after c, 15ms
  E:        e,         after d, 4ms
  F:                   after e, 4ms
{% end %}

---

# Multi-threading

What if multiple nodes could process at the same time? That would reduce the worst-case execution time bound from the amount of time every node in the graph spends processing to the amount of time every node on the [critical path](https://en.wikipedia.org/wiki/Critical_path_method) spends processing (assuming sufficient threads and perfect load balance).

{% mermaid() %}
gantt
  dateFormat SSS
  axisFormat %L
  deadline: milestone, 050,
  A:        a,         000,     4ms
  B:        b,         after a, 27ms
  C:        c,         after a, 8ms
  D:        d,         after b, 15ms
  E:        e,         after c, 4ms
  F:                   after d, 4ms
{% end %}

In reality, we don't have infinite threads at our disposal and don't know beforehand how long each node needs to process, so we'll have to build a good enough schedule during the graph processing step itself.

[^2]: Atomic memory orderings are omitted to simplify the following pseudocode snippets. When in doubt, assume they are `Relaxed`, except where necessary for synchronization.

## Kahn's algorithm

The DFS-based topological sort is ill-suited for parallel processing because it traverses the graph in the reverse order required for processing. The alternative I ended up choosing was a parallel implementation of Kahn's algorithm, which builds a "schedule" on-the-fly, already in the required order. It works as follows:

- initialize each node's *indegree* to the number of direct dependencies it has
- while there are to-process nodes:
    - pick a node with an indegree of 0
    - process that node
    - decrement the indegree of all of its direct dependents

This has the same $\mathcal{O}(|V|+|E|)$ time complexity as the depth-first search based approach, but is naturally parallelizable because all nodes with indegree 0 are logically independent of each other and can therefore be processed simultaneously.

## Attempt 1: Rayon (`spawn`-jobs)

My first attempt at implementing this parallelization was using the `rayon` data parallelism library and its scope API. It quite closely follows the algorithm as described above, spawning a new job into the thread pool for every node whose indegree reaches 0.

```rs
fn run_schedule(nodes: &[Node]) {
    for node in nodes {
        node.indegree.store(node.incoming.len());
    }

    rayon_core::in_place_scope(|s| {
        for node in 0..nodes.len() {
            if nodes[node].indegree.load() == 0 {
                s.spawn(|s| run_node(s, node, nodes))
            }
        }
    });
}

fn run_node<'a>(s: rayon_core::Scope<'a>, node: usize, nodes: &'a [Node]) {
    // "claim" the node to ensure it isn't processed multiple times
    nodes[node].indegree.fetch_sub(1);

    nodes[node].sum_inputs(nodes);
    nodes[node].process();

    for next in &nodes[node].outgoing {
        if nodes[next].indegree.fetch_sub(1) == 1 {
            s.spawn(|s| run_node(s, next, nodes));
        }
    }
}
```

This has a few clear dealbreakers:

1. There's a race condition, can you spot it? [^3]
2. Rayon heap-allocates one `Box<dyn FnOnce()>` per call to `Scope::spawn`.
3. Rayon uses `crossbeam-deque` as its work-stealing queue, which itself heap-allocates job space.
4. Rayon parks the audio thread until all spawned jobs finish.
5. The audio thread doesn't participate in the processing of nodes.

By reserving one dependent node for every job with dependent nodes and tail-calling the worker function for that node, we can save on quite a few job spawns and therefore quite a few `Box<dyn FnOnce()>` allocations:

```diff
+   let mut reserved = None;
+
    for next in &nodes[node].outgoing {
        if nodes[next].indegree.fetch_sub(1) == 1 {
+           if reserved.is_none() {
+               reserved = Some(next);
+           else {
                s.spawn(|s| run_node(s, next, nodes));
+           }
        }
    }
+
+   if let Some(reserved) = reserved {
+       run_node(s, reserved, nodes);
+   }
```

This "reserved node" optimization implements a "work-first" scheduling strategy: the current thread continues processing one dependent node directly while the remaining nodes are made available to other threads.

Additionally, we can at least do *some* processing on the audio thread, by also reserving one node there:

```diff
    rayon_core::in_place_scope(|s| {
+       let mut reserved = None;
+
        for node in 0..nodes.len() {
            if nodes[node].indegree.load() == 0 {
+               if reserved.is_none() {
+                   reserved = Some(node);
+               } else {
                    s.spawn(|s| run_node(s, node, nodes))
+               }
            }
        }
+
+       if let Some(reserved) = reserved {
+           run_node(s, reserved, nodes);
+       }
    });
```

Ideally, this tweak leads to enough subsequent reserved nodes for the audio thread to never have to park, but that can't be guaranteed, so we can't assume it to be the case.

[^3]: This is a classic [TOCTOU](https://en.wikipedia.org/wiki/Time-of-check_to_time-of-use) bug: both the audio thread and a worker thread can independently observe the indegree reaching zero and decide to spawn a job for that node.

## Attempt 2: Rayon (`broadcast`-jobs)

We can get closer to real-time safety if we lower the allocation count even more by distributing the nodes ourselves via a shared lock-free queue.

Rayon lets us run a function on every thread in the pool at the same time with the `broadcast` function, which only allocates for jobs once, regardless of the number of threads. `broadcast`, however, takes a lock, but if we can guarantee that we're the only thread broadcasting jobs, this lock is always uncontended.

Instead of spawning a new job when a node's indegree reaches zero, we just push it into the queue, and each thread pops from the queue until it's empty.

```rs
use crossbeam_queue::ArrayQueue;

fn run_schedule(nodes: &[Node], queue: &ArrayQueue<usize>) {
    for node in nodes {
        node.indegree.store(node.incoming.len());

        if node.incoming.is_empty() {
            _ = queue.push(node);
        }
    }

    rayon_core::broadcast(|_| {
        while let Some(node) = queue.pop() {
            run_node(node, nodes, queue);
        }
    });
}

fn run_node(node: usize, nodes: &[Node], queue: &ArrayQueue<usize>) {
    nodes[node].sum_inputs(nodes);
    nodes[node].process();

    for next in &nodes[node].outgoing {
        if nodes[next].indegree.fetch_sub(1) == 1 {
            _ = queue.push(next);
        }
    }
}
```

This approach, incidentally, also gets rid of the race condition present in the rayon-`spawn`-job-based implementation, since the node "claim" is no longer necessary.

The prior "reserved node" optimization also applies here. However, here it only helps reduce scheduling overhead and contention on the queue, and doesn't actually help towards achieving real-time safety.

```diff
+   let mut reserved = None;
+
    for next in &nodes[node].outgoing {
        if nodes[next].indegree.fetch_sub(1) == 1 {
+           if reserved.is_none() {
+               reserved = Some(next);
+           else {
                _ = queue.push(next);
+           }
        }
    }
+
+   if let Some(reserved) = reserved {
+       run_node(reserved, nodes, queue);
+   }
```

## Attempt 3: Fork-Union

This broadcast-based approach can quite easily be adapted to a different thread pool implementation: `fork_union` is another parallelism library which is focused on low-latency applications and OpenMP-style parallelism, and promises to be real-time safe. Here, it's a near drop-in replacement:

```rs
use crossbeam_queue::ArrayQueue;
use fork_union::ThreadPool;

fn run_schedule(nodes: &[Node], queue: &ArrayQueue<usize>, pool: &mut ThreadPool) {
    /* ... */

    pool.for_threads(|_, _| {
        while let Some(node) = queue.pop() {
            run_node(node, nodes, queue);
        }
    });
}

/* ... */
```

However, at the time of writing, `fork_union`:

1. contains unsound Rust code
2. is written in C++, requiring a C++ compiler
3. keeps worker threads spinning, leading to high CPU usage and power consumption, even outside of the callback time-slice

This approach may be fine for e.g. HPC or Big Data workloads. For consumer situations, however, I consider these trade-offs unacceptable.

## Attempt 4: Writing my own

Since I couldn't find a thread pool implementation that fit my requirements, I decided to give writing my own a try.

The worker threads and the pool itself maintain a few key pieces of shared data:

```rs
struct Shared {
    /// holds both the queue and nodes
    audio_graph: AtomicPtr<AudioGraph>,
    /// increment to wake worker threads
    epoch: AtomicUsize,
    /// usize::MAX if idle, otherwise number of active worker threads
    active: AtomicUsize,
    /// number of remaining nodes to process
    to_do: AtomicUsize,
}
```

Worker threads cycle through three states:

- idle: waiting for `epoch` to change; parks after a short spin
- spinning: polling the queue for nodes to process; returns to idle after a short spin
	- on enter: returns to idle if `active` is `usize::MAX`, otherwise increments `active`
	- on exit: decrements `active`
- working: decrements `to_do`, processes a node, returns to spinning

The audio thread transitions the pool from idle to active by:

1. setting `to_do` to the node count
2. storing the reference to the audio graph
3. resetting `active` from `usize::MAX` to 0
4. incrementing `epoch`
5. unparking worker threads

The audio thread then also participates in the work loop. When it finishes, it decrements `active` and waits until all worker threads stop spinning. The final worker decrements `active` back from 0 to `usize::MAX`, returning the pool to the idle state.

As in the broadcast-job-based implementations, we can reserve one node per thread to reduce scheduling overhead and queue contention. Additionally, giving the audio thread higher priority for new jobs while it spins helps waste less time waiting for workers to finish.

Is this thread-pool implementation real-time safe? Technically no, because the audio thread unparks other threads, which is a system call (`futex_wake` on Linux) with an unknown upper-bound runtime. However, we can't guarantee strict real-time bounds on consumer OSes anyways, and unparking only happens once per callback for a pre-determined number of threads, so the cost is limited in practice, especially on a high-priority thread. Additionally, [SuperCollider's `supernova` takes a similar approach](https://github.com/supercollider/supercollider/blob/3.14/server/supernova/dsp_thread_queue/dsp_thread.hpp#L121).

This algorithm is also not strictly optimal: worker threads leaving the work loop when no new nodes are available may reduce parallelism if a node has multiple dependents. However, my experience producing music says that the [transitive reduction](https://en.wikipedia.org/wiki/Transitive_reduction) of most audio graphs is roughly fan-in-shaped, since they usually follow the structure of tracks → buses → master. In such a scenario it doesn't leave many potential gains on the table.

## Generalizing

To make the thread pool algorithm reusable for other workloads with similar characteristics and requirements, we can separate the scheduler from graph-specific logic via a `WorkList` trait:

```rs
pub trait WorkList: Sync {
    type Item;
    fn next_item(&self) -> Option<Self::Item>;
    fn do_work(&self, item: Self::Item) -> Option<Self::Item>;
}
```

- `next_item` provides the next node to work on
- `do_work` processes it and provides a "reserved" node for immediate processing

The resulting implementation is available under an open-source license in the Generic DAW github repository: [generic-daw/generic-daw](https://github.com/generic-daw/generic-daw/blob/1e195ef/thread_pool/src/lib.rs)

---

# Measurements

Theory is great and all, but to make sure I wasn't massively regressing performance with my own thread pool, I decided to measure *load* (the fraction of the callback time budget used) throughout a test project using the single-threaded scheduler, the rayon-`spawn`-job-based implementation and my own thread pool.

The test project contains 84 processing nodes arranged in an extreme fan-in graph with five layers (71 → 7 → 3 → 2 → 1), which mirrors the patterns I expect to encounter in the wild. It also ensures the scheduler encounters both highly parallelizable regions with many independent nodes and dependency-limited regions where only a few nodes can run simultaneously. Each node contains a running instance of [Spectral Compressor](https://github.com/robbert-vdh/nih-plug/tree/master/plugins/spectral_compressor), which skips processing if its input audio is silent, leading to a very dynamic load distribution between nodes across the test's 38000 measurements.

The test was run on an Intel Core i7-13700H processor, with 6 hyper-threading performance cores at 5 GHz and 8 efficiency cores at 3.7 GHz. The running CPU frequency scaling driver was `intel_pstate`, set to the `performance` power profile, and the running OS was Arch Linux 6.19.6. Audio processing was set to a sample rate of 44.1 kHz and a buffer size of 512 samples, with 18 worker threads in the thread pool.

<div style="display:flex">
	<div style="flex:1">
		<img src="/img/real-time-safe-multi-threaded-daw-audio/histogram_1.svg" style="width:100%" />
	</div>
	<div style="flex:1">
		<img src="/img/real-time-safe-multi-threaded-daw-audio/box_plot_1.svg" style="width:100%" />
	</div>
</div>

First, comparing the single-threaded implementation and both multi-threaded implementations, we can observe a large decrease in load across the board. In fact, both multi-threaded implementations' P100 load is lower than the single-threaded implementation's P75 load, and both multi-threaded implementations' P75 load is lower than the single-threaded implementation's P25 load. Additionally, the single-threaded implementation missed the deadline in 25 measurements, while neither multi-threaded implementation missed the deadline at all.

<div style="display:flex">
	<div style="flex:1">
		<img src="/img/real-time-safe-multi-threaded-daw-audio/histogram_2.svg" style="width:100%" />
	</div>
	<div style="flex:1">
		<img src="/img/real-time-safe-multi-threaded-daw-audio/box_plot_2.svg" style="width:100%" />
	</div>
</div>

Second, comparing the rayon-`spawn`-job-based implementation and my own thread pool, we see load slightly decrease across the board. However, my thread pool showed slightly higher measurement-to-measurement deviation with $\sigma\approx0.0308$, compared to $\sigma\approx0.0283$ for the rayon-`spawn`-job-based implementation. This may be caused by the dynamic nature of the test, since my thread pool has an additional mode around $0.092$ that's not present in the rayon-`spawn`-job-based implementation. Due to the low load at that mode, this indicates less work than average being done, hinting at the overhead of spawning many short jobs dominating the rayon-`spawn`-job-based implementation in that case.

# Conclusion

Real-time audio processing demands predictable, low-latency execution. Multi-threading reduces the time bound from total node processing to the critical path, but introduces challenges like synchronization and scheduling overhead.

Through experimentation - from rayon jobs to a custom thread pool - we’ve seen that careful scheduling, controlled memory use, and letting the audio thread participate in work can eliminate buffer underruns while keeping CPU usage reasonable, making high-performance, real-time audio possible on consumer hardware. In practice, the difficulty of real-time multi-threading is not parallelizing work, but ensuring the scheduler itself has bounded latency.
