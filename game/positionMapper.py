class PositionMapper: 
    
    def fieldNumberFromCoord(self, x, y):
        return (x+8) + (y+8)*17 
    
    
    def fieldNumberFromIdentifier(self, identifier):
        return self.fieldNumbers[identifier]
    
    
    def identifierFromFieldNumber(self, fieldNumber):
        return len([x for x in self.fieldNumbers if x < fieldNumber])
    
    
    def identifierFromCoord(self, x, y):
        return self.identifierFromFieldNumber(self.fieldNumberFromCoord(x, y))
    
    
    def coordFromIdentifier(self, id):
        fieldNumber = self.fieldNumberFromIdentifier(id)
        return self.coordFromFieldNumber(fieldNumber)
    
    
    def coordFromFieldNumber(self, fieldNumber):
        x = (fieldNumber % 17) - 8
        y = (fieldNumber // 17) - 8
        return (x, y)
                        
                    
    def directionMapper(self, di, dj):
        return di + 17 * dj  