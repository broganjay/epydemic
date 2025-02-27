from lark import Lark, Transformer

import json, os

from string import Template

class ModelTransformer(Transformer):

    ## init function inherited 

    def start(self, children):
        return children[0]
    
    def using(self, children):
        return {"imports": {"filename": children[0], "model": children[1]}}
    
    def defs(self, children):
        # return {"defs": children}
        return children

    def model_def(self, children):
        return {"model": {"type": children[0], "name": children[1]} | children[2]}
    
    def MODEL_TYPE(self, children):
        return str(children)
    
    def model_def_body(self, children):
        return children[0] | children[1] | children[2] | children[3]

    def parameters(self, children):
        return {"parameters": children}
    
    def compartments(self, children):
        return {"compartments": children}
    
    def transitions(self, children):
        return {"transitions": children}
    
    def distributions(self, children):
        return {"distribution": children}
    
    def distribution(self, children):
        p = children[1] if len(children) > 1 else -1
        return {"compartment": children[0], "probability": p}
    
    def transition(self, children):
        return children[0]
    
    def edge_transition(self, children):
        return {"type": "edge", "name": children[0], "left_node": children[1], "right_node": children[2], "from": children[3], "to": children[4], "probability": children[5]}
    
    def node_transition(self, children):
        return {"type": "node", "name": children[0], "from": children[1], "to": children[2], "probability": children[3]}

    def NAME(self, children):
        return str(children)

    def TIME(self, children):
        try:
            return float(children)
        except:
            return children
    
    def probability(self, children):
        try:
            p = float("0." + "".join([child for child in children]))
            return p
        except:
            return str(children[0])
        
    def process_def(self, children):
        return {"process": {"inheriting": children[0], "name": children[1]} | children[2]}
    
    def process_def_body(self, children):
        return children[0] | children[1] | children[2] 
    
    def args(self, children):
        return {"args": children}
    
    def arg(self, children):
        return {"name": children[0], "value": children[1]}

    def staging(self, children):
        return {"staging": {"time": children[0]}}
    
    def interactions(self, children):
        return {"interactions": children}
    
    def interaction(self, children):
        return {"type": children[0], "with": children[1]}
    
class ModelBuilder:

    def __init__(self, grammarFile = "def/epsl.lark", outputPath = "gen_models/"):
        self._grammarFile = grammarFile
        with open(self._grammarFile) as f:
            self._grammar = f.read()
        self._parser = Lark(self._grammar, start='start', parser='lalr')
        self._stringsToParse = []
        self._outputPath = outputPath
    
    def addFileToParse(self, filePath):
        with open(filePath) as f:
            self.addStringToParse(f.read())

    def addStringToParse(self, string):
        self._stringsToParse.append(string)

    def buildModels(self, transformer = ModelTransformer()):
        trees = []
        for string in self._stringsToParse:
            trees.append(self._parser.parse(string))
        transformed = []
        for tree in trees:
            transformed.append(transformer.transform(tree))
        all = []
        for t in transformed:
            for mdef in t:
                all.append(mdef)
        self._models = []
        self._processes = []
        for item in all:
            if list(item.keys())[0] == "model":
                self._models.append(item)
            elif list(item.keys())[0] == "process":
                self._processes.append(item)

    def _outputModelSrc(self, model, template: Template, outputPath):
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        substitutions = self._generateSubstitutions(model)
        src = src = template.substitute(substitutions)
        filename = "test"
        with open(outputPath + filename + ".py", "w") as f:
            f.write(src)

    def generateModels(self, outputPath = None):
        if outputPath is None:
            outputPath = self._outputPath 
        with open("def/model_template.tmpl") as f:
            template = Template(f.read())
        for model in self._models:
            self._outputModelSrc(model, template, outputPath)

    def _generateSubstitutions(self, model):
        substitutions = {}
        model = model['model']
        print(model)
        substitutions['name'] = model['name']
        substitutions['params_def_string'] = self._paramsDefString(model['parameters'])
        substitutions['compartments_def_string'] = self._compartmentsDefString(model['compartments'])
        substitutions['edge_def_string'] = self._edgeDefString(model['compartments'])
        substitutions['get_params_string'] = self._getParamsString(model['parameters'])
        substitutions['add_compartments_string'] = self._addCompartmentsString(model['compartments'], model['distribution'])
        print(substitutions)
        return substitutions

    def _paramsDefString(self, parameters):
        string = ""
        for param in parameters:
            string += param.upper() + ": Final[str] = " + "'autogenerated.epydemic." + param + "'"
            string += "\n"
        return string 

    def _compartmentsDefString(self, compartments):
        string = ""
        for c in compartments:
            string += c.upper() + ": Final[str] = " + "'autogenerated.epydemic." + c + "'"
            string += "\n"
        return string 
    
    def _edgeDefString(self, compartments):
        edges = []
        for i in range(len(compartments)):
            for j in range(len(compartments)):
                if not (compartments[j], compartments[i]) in edges:
                    edges.append((compartments[i], compartments[j]))
        string = ""
        for l, r in edges:
            locus_string = l.upper() + r.upper()
            string += locus_string + ": Final[str] = 'autogenerated.epydemic." + locus_string
            string += "\n"
        return string
    
    def _getParamsString(self, params):
        string = ""
        string += "[" + ", ".join(params) + "] = self.getParameters(" + ", ".join(map(lambda s: "self." + s.upper(), params)) + ")"
        return string

    def _addCompartmentsString(self, compartments, distribution):
        otherwiseCompartments = []
        string = ""
        allPs = []
        for entry in distribution:
            print(distribution)
            c = entry['compartment']
            p = entry['probability']
            if p == -1:
                otherwiseCompartments.append(c)
                string += "self.addCompartment(self." + c.upper() + ", " + str(p) + ")"
                string += "\n"
                allPs.append(p)
            else:
                string += "\n"
        for c in otherwiseCompartments:
            p = "1" + " - ".join(map(str, allPs))
            string += "self.addCompartment(self." + c.upper() + ", " + p + ")"
        for c in compartments:
            if c not in distribution
        return string

        

    def __str__(self):
        return "ModelBuilder: " + str(len(self._stringsToParse)) + " models parsed and ready to be built."