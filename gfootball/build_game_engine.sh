#!/bin/bash
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

LIB_EXTENSION="so"
PYTHON_BIN="${PYTHON:-python}"

if [[ "$OSTYPE" == "darwin"* ]] ; then
    LIB_EXTENSION="dylib"
fi

# Take into account # of cores and available RAM for deciding on compilation
# parallelism. If psutil is not available (e.g. isolated build backends), fall
# back to CPU-count based parallelism.
PARALLELISM=$("$PYTHON_BIN" - <<'PY'
import multiprocessing as mp

try:
  import psutil
except ImportError:
  print(max(1, mp.cpu_count()))
else:
  print(int(max(1, min((psutil.virtual_memory().available / 1000000000 - 1) / 0.5,
                       mp.cpu_count()))))
PY
)

# Delete pre-existing version of CMakeCache.txt to make 'python3 -m pip install' work.
rm -f third_party/gfootball_engine/CMakeCache.txt
PYTHON_EXECUTABLE=$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')
pushd third_party/gfootball_engine \
  && cmake . -DPython_EXECUTABLE="$PYTHON_EXECUTABLE" \
  && make -j "$PARALLELISM" \
  && ln -sf "libgame.$LIB_EXTENSION" _gameplayfootball.so \
  && popd
