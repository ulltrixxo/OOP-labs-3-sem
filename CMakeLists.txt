cmake_minimum_required(VERSION 3.24)

project(oop_labs
    VERSION 1.0
    DESCRIPTION "Лабораторные работы по ООП на C++20"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# compile_commands.json нужен language server'у (clangd) для подсказок и диагностики.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Без явного указания конфигурации собираем Debug: в нём включены санитайзеры.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Конфигурация сборки" FORCE)
endif()

option(OOP_ENABLE_SANITIZERS "AddressSanitizer и UndefinedBehaviorSanitizer в Debug" ON)

# Общие настройки компиляции собраны в одну интерфейсную цель: она ничего не
# компилирует сама, а раздаёт флаги всем, кто к ней подключился. Так флаги
# описаны один раз, а не продублированы в каждой цели.
add_library(oop_project_options INTERFACE)

target_compile_options(oop_project_options INTERFACE
    -Wall -Wextra -Wpedantic)

if(OOP_ENABLE_SANITIZERS)
    # $<$<CONFIG:Debug>:...> подставляет флаг только в конфигурации Debug.
    # Санитайзеры нужны и компилятору, и компоновщику — отсюда два вызова.
    target_compile_options(oop_project_options INTERFACE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
        $<$<CONFIG:Debug>:-g>)

    target_link_options(oop_project_options INTERFACE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
endif()

enable_testing()

add_subdirectory(core)
add_subdirectory(apps)
add_subdirectory(tests)
