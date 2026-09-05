#ifndef QWEN38_REASONING_H
#define QWEN38_REASONING_H

#include <stddef.h>
#include <string.h>

typedef struct {
    int open;
    int close_match;
    unsigned ignored_eos;
} Q38ReasoningGuard;

static inline const char *q38_reasoning_last(const char *text, const char *needle)
{
    const char *last = NULL;
    for (const char *hit = strstr(text, needle); hit; hit = strstr(hit + 1, needle))
        last = hit;
    return last;
}

static inline void q38_reasoning_init(Q38ReasoningGuard *guard, const char *prompt)
{
    const char *open = q38_reasoning_last(prompt, "<think>");
    const char *close = q38_reasoning_last(prompt, "</think>");
    memset(guard, 0, sizeof(*guard));
    guard->open = open && (!close || open > close);
}

static inline void q38_reasoning_feed(Q38ReasoningGuard *guard,
                                      const void *bytes, size_t count)
{
    static const char close[] = "</think>";
    const unsigned char *data = bytes;
    if (!guard->open) return;
    for (size_t i = 0; i < count; i++) {
        if (data[i] == (unsigned char)close[guard->close_match])
            guard->close_match++;
        else
            guard->close_match = data[i] == (unsigned char)close[0];
        if (guard->close_match == (int)(sizeof(close) - 1)) {
            guard->open = 0;
            guard->close_match = 0;
            return;
        }
    }
}

static inline int q38_reasoning_ignore_eos(Q38ReasoningGuard *guard)
{
    if (!guard->open) return 0;
    guard->ignored_eos++;
    return 1;
}

static inline int q38_reasoning_budget_reached(const Q38ReasoningGuard *guard,
                                                int budget, int generated)
{
    return guard->open && budget > 0 && generated >= budget;
}

#endif
