#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "../qwen38_tier.h"

int main(void) {
    const size_t gib = (size_t)1 << 30;
    assert(q38t_auto_budget(24 * gib) == 22 * gib);
    assert(q38t_auto_budget(2 * gib) == 0);
    assert(q38t_auto_budget(1 * gib) == 0);
    puts("qwen38 expert budget tests: ok");
    return 0;
}
