#include <assert.h>
#include <stdio.h>

#include "../qwen38_sampler.h"

int main(void) {
    Q38SampleProb candidates[] = {
        {.p = .1f, .id = 30},
        {.p = .5f, .id = 10},
        {.p = .1f, .id = 40},
        {.p = .3f, .id = 20},
    };
    int kept = 0;
    int token = q38_sample_pick(candidates, 4, 1.0, 0, 0.f, .7f, .9, &kept);

    assert(kept == 2);
    assert(token == 20);

    Q38SampleProb constrained[] = {
        {.p = .2f, .id = 30},
        {.p = .05f, .id = 40},
        {.p = .5f, .id = 10},
        {.p = .3f, .id = 20},
    };
    token = q38_sample_pick(constrained, 4, 1.05, 3, .5f, 1.f, .9, &kept);
    assert(kept == 2);
    assert(token == 20);
    puts("qwen38 sampler regression: PASS");
    return 0;
}
