#include "mlcore/matrix_slice.hpp"

template <class T>
MatrixSlice<T>::MatrixSlice(T *p, std::size_t n, std::size_t step, SliceAxis axis)
    : data_(n ? p : nullptr), size_(n), stride_(step), axis_(axis) {
    if (n && (!p || !step))
        throw BadArgument("Invalid slice buffer or stride");
}

template <class T>
std::size_t MatrixSlice<T>::size() const noexcept { return size_; }

template <class T>
bool MatrixSlice<T>::empty() const noexcept { return size_ == 0; }

// шаг
template <class T>
std::size_t MatrixSlice<T>::stride() const noexcept { return stride_; } 

template <class T>
SliceAxis MatrixSlice<T>::axis() const noexcept { return axis_; }

// является ли непрерывной
template <class T>
bool MatrixSlice<T>::is_contiguous() const noexcept { return size_ <= 1 || stride_ == 1; }

template <class T>
T* MatrixSlice<T>::data() const noexcept { return data_; }

// вернуть ссылку на элемент
// учитывается шаг в зависимости от лайаута
template <class T>
T& MatrixSlice<T>::operator()(std::size_t i) const noexcept { return data_[i * stride_]; }

template <class T>
T& MatrixSlice<T>::at(std::size_t i) const {
    if (i >= size_)
        throw std::out_of_range("Slice index");
    return (*this)(i);
}

template <class T>
std::span<T> MatrixSlice<T>::operator*() const {
    return as_span();
}
// вернуть указатель на область памяти
template <class T>
std::span<T> MatrixSlice<T>::as_span() const {
    if (!is_contiguous())
        throw NonContiguousSliceError("Slice is strided; use slice(i)");
    return {data_, size_};
}

template <class T>
MatrixSlice<const double> MatrixSlice<T>::as_const() const noexcept {
    return {data_, size_, stride_, axis_};
}


// создать необходимые реализации
template class MatrixSlice<double>;
template class MatrixSlice<const double>;
