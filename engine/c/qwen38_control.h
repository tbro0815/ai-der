#ifndef QWEN38_CONTROL_H
#define QWEN38_CONTROL_H

#include <string.h>

#include "serve_codec.h"

#define Q38_CONTROL_INTERVAL_S 10.0

typedef enum {
    Q38_CONTROL_NONE = 0,
    Q38_CONTROL_STOP,
    Q38_CONTROL_CANCEL,
    Q38_CONTROL_NOT_FOUND,
    Q38_CONTROL_BUSY,
} Q38ControlAction;

static inline Q38ControlAction q38_control_action(
    ColiServeCommandKind kind, const char *id, const char *active)
{
    if (kind == COLI_SERVE_COMMAND_SUBMIT || kind == COLI_SERVE_COMMAND_RESET)
        return Q38_CONTROL_BUSY;
    if (kind != COLI_SERVE_COMMAND_STOP && kind != COLI_SERVE_COMMAND_CANCEL)
        return Q38_CONTROL_NONE;
    if (!active || strcmp(id, active)) return Q38_CONTROL_NOT_FOUND;
    return kind == COLI_SERVE_COMMAND_CANCEL ? Q38_CONTROL_CANCEL : Q38_CONTROL_STOP;
}

#endif
