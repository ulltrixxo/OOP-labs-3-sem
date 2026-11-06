#pragma once
#include <stdexcept>
#include <format>
#include <source_location>
#include <string>
#include <string_view>

namespace diagnostics {

// The default argument captures the message() call site, i.e. the throw location.
/** @brief Add source filename, line and function to a diagnostic.
 * @param reason Error description.
 * @param location Defaults to the call site, not this helper body. */
inline std::string message(
    std::string_view reason,
    std::source_location location = std::source_location::current())
{
    return std::format("{}\n  Source: {}:{}\n  Function: {}",
                       reason, location.file_name(), location.line(),
                       location.function_name());
}

} // namespace diagnostics

/** @brief Base for invalid caller input. */
class ArgumentError : public std::invalid_argument {
    public:
        using std::invalid_argument::invalid_argument;
};

/** @brief Incompatible dimensions or target structure. */
class SizeError : public ArgumentError {
    public:
        using ArgumentError::ArgumentError;
};

/** @brief Invalid parameter value. */
class BadArgument : public ArgumentError {
    public:
        using ArgumentError::ArgumentError;
};


/** @brief Prediction requested before successful fitting. */
class ModelFitError : public std::logic_error {
    public:
        using std::logic_error::logic_error;
};

/** @brief Unsupported matrix layout. */
class MatrixTypeError : public std::logic_error {
    public: using std::logic_error::logic_error;
};


/** @brief Base for file failures. */
class FileError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
};

/** @brief Input file could not be opened. */
class FileOpenError : public FileError {
    public:
        using FileError::FileError;
};

/** @brief Input read failed or binary contents are invalid. */
class FileReadError : public FileError {
    public:
        using FileError::FileError;
};

/** @brief Text input has an invalid structure. */
class FileFormatError : public FileError {
    public:
        using FileError::FileError;
};

/** @brief Output write failure. */
class FileWriteError : public FileError {
    public:
        using FileError::FileError;
};

/** @brief Binary save failed. */
class FileSaveError : public FileError {
    public:
        using FileError::FileError;
};