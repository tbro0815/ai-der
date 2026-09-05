#include <assert.h>
#include <stdio.h>

#include "../qwen38_reasoning.h"

int main(void)
{
    Q38ReasoningGuard guard;
    q38_reasoning_init(&guard, "old <think>done</think> next <think>");
    assert(guard.open);
    assert(q38_reasoning_ignore_eos(&guard));
    assert(guard.ignored_eos == 1);
    assert(!q38_reasoning_budget_reached(&guard, 0, 8192));
    assert(!q38_reasoning_budget_reached(&guard, 8192, 8191));
    assert(q38_reasoning_budget_reached(&guard, 8192, 8192));

    q38_reasoning_feed(&guard, "work</thi", 9);
    assert(guard.open);
    q38_reasoning_feed(&guard, "nk>answer", 9);
    assert(!guard.open);
    assert(!q38_reasoning_budget_reached(&guard, 8192, 9000));
    assert(!q38_reasoning_ignore_eos(&guard));
    assert(guard.ignored_eos == 1);

    q38_reasoning_init(&guard, "<think></think>");
    assert(!guard.open);
    puts("qwen38 reasoning guard tests: ok");
    return 0;
}
