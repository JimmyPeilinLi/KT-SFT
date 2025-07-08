#ifndef SFT_DEBUG_HPP
#define SFT_DEBUG_HPP

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>

inline std::string get_env_or_default(const char *var_name, const std::string &default_value) {
  const char *value = std::getenv(var_name);
  return (value != nullptr) ? std::string(value) : default_value;
}

// inline void dump_bin(std::string file_name, float16_t *data, size_t count) {
//   file_name = get_env_or_default("SFT_DEBUG_PATH", "debug") + "/" + file_name + ".f16";
//   std::ofstream f(file_name, std::ios::binary);
//   f.write(reinterpret_cast<const char *>(data), count * sizeof(*data));
//   f.close();
// }
inline void dump_bin(std::string file_name, float *data, size_t count) {
  file_name = get_env_or_default("SFT_DEBUG_PATH", "debug") + "/" + file_name + ".f32";
  std::cout << file_name << std::endl;
  std::ofstream f(file_name, std::ios::binary);
  f.write(reinterpret_cast<const char *>(data), count * sizeof(*data));
  f.close();
}
inline void dump_bin(std::string file_name, int64_t *data, size_t count) {
  file_name = get_env_or_default("SFT_DEBUG_PATH", "debug") + "/" + file_name + ".int64";
  std::cout << file_name << std::endl;
  std::ofstream f(file_name, std::ios::binary);
  f.write(reinterpret_cast<const char *>(data), count * sizeof(*data));
  f.close();
}
inline void dump_bin(std::string file_name, uint8_t *data, size_t count) {
  file_name = get_env_or_default("SFT_DEBUG_PATH", "debug") + "/" + file_name + ".uint8";
  std::cout << file_name << std::endl;
  std::ofstream f(file_name, std::ios::binary);
  f.write(reinterpret_cast<const char *>(data), count * sizeof(*data));
  f.close();
}

#endif
