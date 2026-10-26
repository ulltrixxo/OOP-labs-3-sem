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

# Обязательные проверки условия работы 1 и где они лежат:
#   разбор строки с пропуском         — test_loader
#   определение типа признака         — test_loader
#   расчёт квантилей на известном массиве — test_calculate
#   корректность кодирования категорий — test_encoding
#   цикл «запись — чтение» бинарного файла — test_binary
#   реакция на несогласованный CSV    — test_loader
oop_add_test(test_loader)
oop_add_test(test_calculate)
oop_add_test(test_encoding)
oop_add_test(test_binary)

# Работа 2
oop_add_test(test_split)

# Отдельные программы для проверок, которые при ошибке в ядре аварийно
# останавливают процесс: так они не обрывают остальные тесты.
oop_add_test(test_calculate_small)
oop_add_test(test_binary_errors)
