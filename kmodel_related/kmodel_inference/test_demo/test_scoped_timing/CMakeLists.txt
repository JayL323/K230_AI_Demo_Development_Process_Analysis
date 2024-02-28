set(src main.cc)
set(bin test_scoped_timing.elf)

add_executable(${bin} ${src})
install(TARGETS ${bin} DESTINATION bin)