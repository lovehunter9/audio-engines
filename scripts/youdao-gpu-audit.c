#define _GNU_SOURCE
#include <link.h>
#include <string.h>

unsigned int la_version(unsigned int version) {
    return version < LAV_CURRENT ? version : LAV_CURRENT;
}

char *la_objsearch(const char *name, uintptr_t *cookie, unsigned int flag) {
    (void)cookie;
    (void)flag;
    if (strcmp(name, "/usr/local/vgpu/libvgpu.so") == 0)
        return "/app/wrapper/s2st/gpu-client/libvgpu.so";
    return (char *)name;
}
