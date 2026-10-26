# Библиотека ядра: загрузка данных, статистика, преобразования, модели.
# Про ввод-вывод для пользователя и аргументы командной строки она не знает —
# это ответственность приложений в apps/.

# GLOB перечисляет исходники автоматически: добавленный в src/ файл попадает в
# сборку без правки этого списка. CONFIGURE_DEPENDS заставляет CMake проверять
# каталог перед каждой сборкой, иначе новый файл увидится только после ручного
# перезапуска cmake.
file(GLOB_RECURSE OOP_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

if(OOP_CORE_SOURCES)
    add_library(oop_core STATIC ${OOP_CORE_SOURCES})

    # PUBLIC: требования нужны и при сборке самой библиотеки, и тем, кто её
    # подключает.
    set(OOP_CORE_SCOPE PUBLIC)
else()
    # Пока в src/ нет ни одного .cpp, компилировать нечего: статическая
    # библиотека без исходников — ошибка CMake. До появления первой реализации
    # ядро существует как header-only, то есть только раздаёт заголовки.
    add_library(oop_core INTERFACE)

    # У INTERFACE-цели нет собственной сборки, поэтому все требования только
    # передаются дальше.
    set(OOP_CORE_SCOPE INTERFACE)

    message(STATUS "oop_core: в core/src нет .cpp, цель собрана как header-only")
endif()

# Каталог заголовков виден как #include "mlcore/....hpp".
target_include_directories(oop_core ${OOP_CORE_SCOPE}
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(oop_core ${OOP_CORE_SCOPE} oop_project_options)
