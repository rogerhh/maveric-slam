print("const int8_t count_lookup[256] = {", end="")
for i in range(256):
    print(f"{bin(i).count('1')}, ", end="")
print("};\n")


