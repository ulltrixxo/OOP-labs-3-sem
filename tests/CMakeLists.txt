# Каждый тест — отдельная программа, возвращающая ненулевой код при провале
# проверки. Отдельный фреймворк не используется, хватает testing.hpp.
function(oop_add_test name)
    add_executable(${name} ${name}.cpp)

    target_link_libraries(${name} PRIVATE oop_core)

    # Чтобы тест писал #include "testing.hpp" без относительных путей.
    target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

    add_test(NAME ${name} COMMAND ${name})

    # UBSan по умолчанию печатает сообщение и продолжает работу, из-за чего тест
    # завершился бы успешно при неопределённом поведении. halt_on_error делает
    # такое срабатывание падением, и CTest его замечает.
    set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1")
endfunction()

oop_add_test(test_smoke)
oop_add_test(test_calculate)
