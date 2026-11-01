# Third-party notices

Semantic DB Search is © 2026 Manuel Dzimah and released under the MIT License (see `LICENSE`).
The Windows app is distributed together with the open-source components listed below.

Licence identifiers and texts were taken from each component's installed package metadata (the
`.dist-info` folders and licence files of the versions pinned in `requirements.txt`), from the
model card at the pinned model revision, and from the Lucide repository. In the installed app the
complete licence texts are in the `_internal\licenses` folder next to `SemanticDBSearch.exe`
(collected by `packaging/semdb.spec`); texts that the packages do not ship themselves are kept in
`packaging/licenses/`.

| Component | Version | Licence | Copyright |
|---|---|---|---|
| [Python](https://www.python.org) (CPython runtime and standard library) | 3.13 | PSF-2.0 (bundled libraries: `licenses/Python/LICENSE.txt`) | Python Software Foundation |
| [Qt 6](https://www.qt.io) and [Qt for Python](https://pyside.org) (PySide6-Essentials, Shiboken6) | 6.12.0 | LGPL-3.0-only as used here (offered as LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only) | The Qt Company Ltd. and contributors |
| [ONNX Runtime](https://onnxruntime.ai) | 1.31.1 | MIT (compiled-in components: `licenses/onnxruntime/ThirdPartyNotices.txt`) | Microsoft Corporation |
| [NumPy](https://numpy.org) | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0; bundled OpenBLAS and LAPACK (BSD-3-Clause), GCC runtime (GPL-3.0-or-later WITH GCC-exception-3.1) | NumPy Developers |
| [Tokenizers](https://github.com/huggingface/tokenizers) | 0.23.3 | Apache-2.0 (compiled-in Rust crates, mostly MIT OR Apache-2.0: `licenses/tokenizers/*.cyclonedx.json`) | Hugging Face (Nicolas Patry, Anthony Moi) |
| [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) | 3.14.6 | MIT | Max Bachmann; Adam Cohen |
| [python-calamine](https://github.com/dimastbk/python-calamine) | 0.9.0 | MIT (compiled-in Rust crates: `licenses/python-calamine/*.cyclonedx.json`) | dimastbk |
| [XlsxWriter](https://github.com/jmcnamara/XlsxWriter) | 3.3.1 | BSD-2-Clause | John McNamara |
| [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) embedding model (modified: quantized) | revision `e8f8c21` | Apache-2.0 | Nils Reimers, Iryna Gurevych (sentence-transformers) |
| [Lucide](https://lucide.dev) icons | — | ISC; icons derived from Feather: MIT | Lucide Icons and Contributors; Cole Bemis |
| [PyInstaller](https://pyinstaller.org) bootloader | 6.22.3 | GPL-2.0-or-later with the Bootloader Exception | PyInstaller Development Team; Giovanni Bajo; McMillan Enterprises, Inc. |
| Microsoft Visual C++ runtime DLLs | 14.x | Microsoft Visual C++ redistributable terms | Microsoft Corporation |

## Qt and Qt for Python (LGPL-3.0)

This application uses the Qt 6 libraries and Qt for Python (PySide6, Shiboken6) under the GNU
Lesser General Public License version 3.

* Qt and PySide6 are used **unmodified** and are **dynamically linked**: they ship as separate,
  replaceable files in the `_internal` folder of the installation (`Qt6Core.dll`, `Qt6Gui.dll`,
  `Qt6Widgets.dll`, `Qt6Svg.dll`, the Qt plugins under `PySide6\plugins`, the `PySide6\*.pyd` and
  `shiboken6\*.pyd` modules, `pyside6.abi3.dll` and `shiboken6.abi3.dll`). You may replace them
  with your own compatible builds of the same Qt / PySide6 version, and you may reverse-engineer
  this application as far as needed to debug such modifications.
* The complete source code of Qt 6.12.0 and Qt for Python 6.12.0 is published by the Qt Project:
  <https://download.qt.io/official_releases/qt/>, <https://download.qt.io/official_releases/QtForPython/>
  and <https://code.qt.io/cgit/pyside/pyside-setup.git/>.
* Copies of the GNU LGPL v3 and of the GNU GPL v3 (which the LGPL incorporates) are in
  `licenses/Qt/LGPL-3.0.txt` and `licenses/Qt/GPL-3.0.txt`.
* Qt contains third-party components with their own licences; they are listed in
  "Licenses Used in Qt": <https://doc.qt.io/qt-6/licenses-used-in-qt.html>.

## ONNX Runtime (MIT)

```text
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

ONNX Runtime is built with further open-source components; their notices are reproduced in
`licenses/onnxruntime/ThirdPartyNotices.txt`.

## NumPy (BSD-3-Clause)

```text
Copyright (c) 2005-2025, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

This binary distribution of NumPy also bundles OpenBLAS (BSD-3-Clause), LAPACK
(BSD-3-Clause-Open-MPI) and the GCC runtime library (GPL-3.0-or-later WITH GCC-exception-3.1),
plus smaller components under 0BSD, MIT, Zlib and CC0-1.0. Their full notices are in
`licenses/numpy/`.

## Tokenizers (Apache-2.0)

Hugging Face Tokenizers is licensed under the Apache License, Version 2.0
(`licenses/tokenizers/Apache-2.0.txt`). Its compiled extension contains Rust crates under
permissive licences (mostly "MIT OR Apache-2.0"), listed with their licences in the CycloneDX
software bill of materials `licenses/tokenizers/tokenizers-python.cyclonedx.json`.

## Embedding model (Apache-2.0)

`models/paraphrase-multilingual-MiniLM-L12-v2` is derived from
[sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
at revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, licensed under the Apache License,
Version 2.0 (`licenses/paraphrase-multilingual-MiniLM-L12-v2/Apache-2.0.txt`).

**Modification notice:** `model.onnx` is not the original file. `tools/fetch_model.py` downloads
the original `onnx/model.onnx` and `tokenizer.json`, verifies their SHA-256 hashes and converts
the model weights to 8-bit integers with ONNX Runtime dynamic quantization. `tokenizer.json` is
unchanged. `manifest.json` next to the model records the source revision, the hashes and the
quantization settings.

The model was introduced in: Nils Reimers and Iryna Gurevych, "Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks", Proceedings of the 2019 Conference on Empirical Methods in Natural
Language Processing, 2019.

## RapidFuzz (MIT)

```text
Copyright © 2020-present Max Bachmann
Copyright © 2011 Adam Cohen

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## python-calamine (MIT)

```text
MIT License

Copyright (c) 2021 dimastbk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Its compiled extension contains Rust crates (among them calamine) under permissive licences,
listed in `licenses/python-calamine/python-calamine.cyclonedx.json`.

## XlsxWriter (BSD-2-Clause)

```text
BSD 2-Clause License

Copyright (c) 2013-2026, John McNamara <jmcnamara@cpan.org>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Lucide icons (ISC)

The interface icons are adapted from Lucide (<https://lucide.dev>).

```text
ISC License

Copyright (c) 2026 Lucide Icons and Contributors

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
```

Lucide icons derived from the Feather project are also covered by:

```text
The MIT License (MIT)

Copyright (c) 2013-present Cole Bemis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The full licence file, including the list of Feather-derived icons, is
`licenses/Lucide/Lucide-ISC.txt`.

## Python (PSF-2.0)

The application runs on an embedded copy of the CPython 3.13 runtime and parts of its standard
library, licensed under the Python Software Foundation License Version 2: "Copyright (c)
2001-2024 Python Software Foundation; All Rights Reserved". The interpreter ships with libraries
under their own licences (among them OpenSSL, SQLite, libffi, zlib, bzip2, XZ Utils and expat).
The PSF licence, the history of the software, the licences of those libraries and the conditions
for redistributing the Microsoft Distributable Code linked into the Windows build are reproduced
in `licenses/Python/LICENSE.txt`.

## PyInstaller bootloader (GPL-2.0-or-later with Bootloader Exception)

`SemanticDBSearch.exe` starts with the PyInstaller bootloader, which may be distributed with
programs under any licence thanks to this exception (full terms in
`licenses/PyInstaller/COPYING.txt`):

```text
================================
 The PyInstaller licensing terms
================================
 

Copyright (c) 2010-2023, PyInstaller Development Team
Copyright (c) 2005-2009, Giovanni Bajo
Based on previous work under copyright (c) 2002 McMillan Enterprises, Inc.


PyInstaller is licensed under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2 of the License,
or (at your option) any later version.


Bootloader Exception
--------------------

In addition to the permissions in the GNU General Public License, the
authors give you unlimited permission to link or embed compiled bootloader
and related files into combinations with other programs, and to distribute
those combinations without any restriction coming from the use of those
files. (The General Public License restrictions do apply in other respects;
for example, they cover modification of the files, and distribution when
not linked into a combined executable.)
 
 
Bootloader and Related Files
----------------------------
```

## Microsoft Visual C++ runtime

`vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140*.dll` and related files are Microsoft
Distributable Code, redistributed unmodified with the Python and Qt binaries under Microsoft's
Visual C++ redistribution terms (see "Additional Conditions for this Windows binary build" in
`licenses/Python/LICENSE.txt`).
