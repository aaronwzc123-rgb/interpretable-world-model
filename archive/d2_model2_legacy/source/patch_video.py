import io

with io.open("tools.py", "r", encoding="utf-8") as f:
    text = f.read()

old = "            self._writer.add_video(name, value, step, 16)"
new = "            pass  # add_video disabled (numpy newshape incompatibility)"

if old in text:
    with io.open("tools.py", "w", encoding="utf-8") as f:
        f.write(text.replace(old, new))
    print("PATCHED: add_video disabled")
elif new in text:
    print("ALREADY PATCHED")
else:
    print("LINE NOT FOUND - paste me the indentation")
