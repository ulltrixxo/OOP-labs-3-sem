#pragma once

// Минимальная обвязка для тестов: проверки считают ошибки, а main возвращает
// ненулевой код, если хотя бы одна из них провалилась. Этого достаточно для
// регистрации теста в CTest.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace testing
{

inline int failed = 0;
inline int passed = 0;

inline void report(bool ok, std::string_view expression,
                   std::string_view file, int line)
{
    if (ok)
    {
        ++passed;
        return;
    }

    ++failed;
    std::cerr << file << ':' << line << ": провалена проверка: "
              << expression << '\n';
}

inline void report_near(double actual, double expected, double tolerance,
                        std::string_view expression,
                        std::string_view file, int line)
{
    const bool ok = std::abs(actual - expected) <= tolerance;

    if (!ok)
    {
        std::cerr << file << ':' << line << ": ожидалось " << expected
                  << ", получено " << actual
                  << " (допуск " << tolerance << ")\n";
    }

    report(ok, expression, file, line);
}

// Возвращается из main: 0, если все проверки прошли.
inline int summary()
{
    std::cout << "проверок пройдено: " << passed
              << ", провалено: " << failed << '\n';

    return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace testing

#define CHECK(expression) \
    ::testing::report(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

#define CHECK_NEAR(actual, expected, tolerance) \
    ::testing::report_near((actual), (expected), (tolerance), \
                           #actual " ~= " #expected, __FILE__, __LINE__)

// Проверяет, что выражение бросило исключение ожидаемого типа. Понадобится
// начиная с работы 2, где об ошибках сообщают исключениями.
#define CHECK_THROWS_AS(expression, exception_type)                          \
    do                                                                       \
    {                                                                        \
        bool thrown = false;                                                 \
        try                                                                  \
        {                                                                    \
            (void) (expression);                                             \
        }                                                                    \
        catch (const exception_type&)                                        \
        {                                                                    \
            thrown = true;                                                   \
        }                                                                    \
        catch (...)                                                          \
        {                                                                    \
        }                                                                    \
        ::testing::report(thrown,                                            \
                          #expression " бросает " #exception_type,           \
                          __FILE__, __LINE__);                               \
    } while (false)
