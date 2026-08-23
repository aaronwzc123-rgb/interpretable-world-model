text = open("envs/__init__.py", encoding="utf-8").read()
anchor = (
    '    elif suite == "crafter":\n'
    "        import envs.crafter as crafter\n"
    "\n"
    "        env = crafter.Crafter(task, config.size, seed=config.seed + id)\n"
    "        env = wrappers.OneHotAction(env)\n"
)
block = (
    '    elif suite == "minigrid":\n'
    "        import envs.minigrid as minigrid\n"
    "\n"
    "        env = minigrid.MiniGrid(task, config.size, seed=config.seed + id)\n"
    "        env = wrappers.OneHotAction(env)\n"
)
if anchor not in text:
    print("ANCHOR NOT FOUND")
elif 'suite == "minigrid"' in text:
    print("ALREADY INSERTED")
else:
    open("envs/__init__.py", "w", encoding="utf-8").write(text.replace(anchor, anchor + block))
    print("minigrid branch inserted OK")
