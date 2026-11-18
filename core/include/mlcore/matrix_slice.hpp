#pragma once
#include "mlcore/exceptions.hpp"
#include <span>
#include <type_traits>

class NonContiguousSliceError : public MatrixTypeError {
  public:
    using MatrixTypeError::MatrixTypeError;
};
enum class SliceAxis { Row, Column };


template <class T> class MatrixSlice {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, const double>); // только double
    T* data_ = nullptr;
    std::size_t size_ = 0, stride_ = 1;
    SliceAxis axis_ = SliceAxis::Row;

  public:
    MatrixSlice() noexcept = default;
    MatrixSlice(T *p, std::size_t n, std::size_t step, SliceAxis axis);
    std::size_t size() const noexcept;
    bool empty() const noexcept;
    std::size_t stride() const noexcept;
    SliceAxis axis() const noexcept;
    bool is_contiguous() const noexcept; // лежат ли элеметы подряд
    T* data() const noexcept;
    T& operator()(std::size_t i) const noexcept;
    T& at(std::size_t i) const;
    std::span<T> operator*() const;
    std::span<T> as_span() const;
    MatrixSlice<const double> as_const() const noexcept;
};

