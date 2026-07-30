class FieldPositionsMapper:
    """Two-way lookup between a field's ``id`` and its ``coord``.

    The board itself is indexed by id, so this exists for the callers that
    start from a coordinate instead — the visualization projecting and
    hit-testing clicks, and ``Move`` locating the field jumped over.
    """

    def __init__(self):
        self.coordById = {}
        self.idByCoord = {}


    def addFieldPosition(self, field):
        self.coordById[field.id] = field.coord
        self.idByCoord[field.coord] = field.id


    def coordFromId(self, id):
        return self.coordById[id]


    def idFromCoord(self, coord):
        return self.idByCoord[coord]
