class FieldPositionsMapper:    
    def __init__(self):
        self.fieldPositions = []
        self.coordDict = {}
        self.idDict	= {}
        self.fieldNumberDict = {}
        
    def addFieldPosition(self, position):
        coord, id, fieldNumber = position["coord"], position["id"], position["fieldNumber"]
        self.fieldPositions.append(position)
        self.idDict[id] = {"coord": coord, "fieldNumber": fieldNumber}
        self.fieldNumberDict[fieldNumber] = {"id": id, "coord": coord}
        self.coordDict[coord] = {"id": id, "fieldNumber": fieldNumber}
        
    
    def coordFromId(self, id):
        return self.idDict[id]["coord"]
    
    
    def idFromCoord(self, coord):
        return self.coordDict[coord]["id"]