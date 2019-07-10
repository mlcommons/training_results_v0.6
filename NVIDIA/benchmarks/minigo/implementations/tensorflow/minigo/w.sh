#!/bin/bash

"${@}" |& tee eval_error.log

