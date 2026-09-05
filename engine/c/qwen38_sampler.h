#ifndef QWEN38_SAMPLER_H
#define QWEN38_SAMPLER_H

typedef struct { float p; int id; } Q38SampleProb;

static int q38_sample_greater(Q38SampleProb a, Q38SampleProb b) {
    return a.p > b.p || (a.p == b.p && a.id < b.id);
}

static void q38_sample_siftdown(Q38SampleProb *heap, int n, int i) {
    Q38SampleProb value = heap[i];
    for (;;) {
        int left = 2 * i + 1;
        if (left >= n) break;
        int best = left;
        if (left + 1 < n && q38_sample_greater(heap[left + 1], heap[left])) best++;
        if (!q38_sample_greater(heap[best], value)) break;
        heap[i] = heap[best];
        i = best;
    }
    heap[i] = value;
}

static Q38SampleProb q38_sample_pop(Q38SampleProb *heap, int *n) {
    Q38SampleProb root = heap[0];
    heap[0] = heap[--*n];
    if (*n) q38_sample_siftdown(heap, *n, 0);
    heap[*n] = root;
    return root;
}

static int q38_sample_pick(Q38SampleProb *heap, int n, double total,
                           int top_k, float min_p, float top_p, double unit_random,
                           int *kept_out) {
    int constrained = (top_k > 0 && top_k < n) || min_p > 0.f;
    if (!constrained && !(top_p > 0.f && top_p < 1.f)) {
        double threshold = unit_random * total, cumulative = 0;
        int token = heap[0].id;
        for (int i = 0; i < n; i++) {
            cumulative += heap[i].p;
            if (cumulative >= threshold) { token = heap[i].id; break; }
        }
        if (kept_out) *kept_out = n;
        return token;
    }
    for (int i = n / 2 - 1; i >= 0; i--) q38_sample_siftdown(heap, n, i);
    int heap_n = n, allowed = 0;
    double mass = total;
    if (constrained) {
        int limit = top_k > 0 && top_k < n ? top_k : n;
        float minimum = min_p > 0.f ? min_p * heap[0].p : 0.f;
        mass = 0;
        while (allowed < limit && heap_n && (!allowed || heap[0].p >= minimum)) {
            mass += q38_sample_pop(heap, &heap_n).p;
            allowed++;
        }
    }
    double target = top_p > 0.f && top_p < 1.f ? top_p * mass : mass;
    double kept = 0;
    int count = 0;
    if (allowed) {
        while (count < allowed && kept < target) kept += heap[n - 1 - count++].p;
    } else {
        while (heap_n && kept < target) { kept += q38_sample_pop(heap, &heap_n).p; count++; }
    }
    double threshold = unit_random * kept, cumulative = 0;
    int token = heap[n - 1].id;
    for (int i = 0; i < count; i++) {
        Q38SampleProb item = heap[n - 1 - i];
        cumulative += item.p;
        if (cumulative >= threshold) { token = item.id; break; }
    }
    if (kept_out) *kept_out = count;
    return token;
}

#endif
