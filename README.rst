nice_svm
========

Wrapper around libsvm to make it easier to use and make the code nicer.

This wrapper tries to make it more C++-like (RAII and use of std::vector),
instead of the very C-like interface provided by libsvm.

This wrapper does not add any feature, it is just an interface to libsvm.
Moreover, this repository does not contain libsvm, you have to install it on
your computer or add it directly to your project.

License
-------

The nice-svm source code is available under the terms of the MIT license, see
`LICENSE` for details.
