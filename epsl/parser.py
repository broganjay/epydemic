from model_builder import ModelBuilder

m = ModelBuilder()
m.addFileToParse("examples/diseases.epsl")
m.addFileToParse("examples/sir.epsl")
m.buildModels()
m.generateModels()