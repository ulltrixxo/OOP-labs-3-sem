# Консольный вывод, общий для приложений всех работ: сводка по признакам,
# статистика, таблицы предсказаний. Это не часть ядра — ядро не печатает, —
# а отдельная маленькая библиотека, которую подключают приложения.
add_library(oop_console STATIC visualize.cpp)

# Приложения подключают заголовок как #include "visualize.hpp".
target_include_directories(oop_console PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(oop_console PUBLIC oop_core)
