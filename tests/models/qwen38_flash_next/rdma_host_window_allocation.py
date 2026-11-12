#!/usr/bin/env python3
"""Exercise actual host-window helpers with real mmap and a fake HIP registrar.

No GPU, RDMA provider, root or system changes. This is allocation/error cleanup
coverage only; native transport/GPU validation is documented separately.
"""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[3]
source = (root / 'src/models/qwen38_flash_next/kernels/rocm/verbs.cpp').read_text()
begin = source.index('bool MapHostBuffer(')
end = source.index('/// The immediate of a GPU-window exchange', begin)
helpers = source[begin:end]
assert helpers.count('bool MapHostBuffer(') == 1
assert helpers.count('void UnmapHostBuffer(') == 1
prefix = r'''
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
void Need(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
constexpr int hipSuccess=0, hipHostRegisterMapped=2;
bool fail_registration=false;
unsigned registers=0,unregisters=0;
void* last_registration=nullptr;
void SetError(std::string* out,std::string message) { if(out)*out=std::move(message); }
int hipHostRegister(void* p,std::size_t,int flags) {
 Need(flags==hipHostRegisterMapped,"registration flags");
 ++registers;last_registration=p;return fail_registration?1:hipSuccess;
}
int hipHostUnregister(void*) { ++unregisters;return hipSuccess; }
'''
suffix = r'''
void NeedUnmapped(void* p) {
 unsigned char state=0;errno=0;
 Need(mincore(p,4096,&state)==-1&&errno==ENOMEM,"mapping leaked");
}
int main() {
 try {
  auto* legacy=reinterpret_cast<void*>(std::uintptr_t{0x700000000000ULL});
  void* held=mmap(legacy,4096,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE,-1,0);
  Need(held==legacy||(held==MAP_FAILED&&errno==EEXIST),"safe legacy reservation");
  void* collision=mmap(legacy,8192,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE,-1,0);
  Need(collision==MAP_FAILED&&errno==EEXIST,"legacy failure control");
  std::string error;void* first=nullptr;void* second=nullptr;
  Need(MapHostBuffer(8192,&first,&error)&&first&&first!=legacy,"first unused mapping");
  Need(MapHostBuffer(8192,&second,&error)&&second&&second!=first&&second!=legacy,"second unused mapping");
  *static_cast<volatile unsigned char*>(first)=17;
  *static_cast<volatile unsigned char*>(second)=29;
  Need(*static_cast<volatile unsigned char*>(first)==17,"distinct usable mappings");
  Need(registers==2&&unregisters==0,"successful registration count");
  UnmapHostBuffer(first,8192);NeedUnmapped(first);
  UnmapHostBuffer(second,8192);NeedUnmapped(second);
  Need(unregisters==2,"successful unregister count");
  UnmapHostBuffer(nullptr,8192);Need(unregisters==2,"null cleanup");
  fail_registration=true;void* failed=nullptr;error.clear();
  Need(!MapHostBuffer(8192,&failed,&error)&&!failed&&!error.empty(),"registration error propagated");
  Need(registers==3&&unregisters==2,"failed registration not unregistered");
  NeedUnmapped(last_registration);
  void* invalid=nullptr;error.clear();
  Need(!MapHostBuffer(0,&invalid,&error)&&!invalid&&!error.empty(),"mmap error propagated");
  Need(registers==3,"failed mmap not registered");
  if(held==legacy)Need(munmap(held,4096)==0,"reservation cleanup");
  std::cout<<"PASS occupied-address/control, two distinct mappings, null/success cleanup, registration/mmap failures\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
'''
with tempfile.TemporaryDirectory(prefix='gufo-rdma-host-window-') as directory:
    folder = Path(directory)
    fixture = folder / 'fixture.cpp'
    binary = folder / 'fixture'
    fixture.write_text(prefix + helpers + suffix)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    subprocess.run([*compiler, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', str(fixture), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)
print('Allocation helpers compiled verbatim from production; HIP registration is simulated, not GPU/RDMA evidence.')
