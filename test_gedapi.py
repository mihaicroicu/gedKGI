from gedAPI import GedLoader

ged = GedLoader(version='21.1', verbose=True)
print(ged.exists)