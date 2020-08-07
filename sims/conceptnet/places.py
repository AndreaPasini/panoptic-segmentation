import re
import json
import pandas as pd
from networkx import MultiDiGraph
from config import places_json_path, conceptnet_coco_places_csv_path
from sims.conceptnet.utils import get_readable_concepts, get_coco_concepts_map, coco_to_concept


class Conceptnet:
    df_conceptnet = None        # Dataframe with conceptnet relationships
    coco_concepts_map = None    # Conceptnet map (coco->concept)
    places = None               # Set of places

    def __init__(self):
        df_conceptnet = get_readable_concepts(
            pd.read_csv(conceptnet_coco_places_csv_path, sep='\t', header=None))
        meaningful_rel = {'Antonym', 'RelatedTo', 'AtLocation', 'IsA', 'HasA', 'PartOf', 'UsedFor', 'MadeOf',
                          'LocatedNear'}
        self.df_conceptnet = df_conceptnet.loc[lambda x: x[1].isin(meaningful_rel)]
        self.coco_concepts_map = get_coco_concepts_map()
        with open(places_json_path) as f:
            places_json = json.load(f)
        self.places = places_json['places'] + places_json['sub-places']

    def get_related_concepts(self, concepts, antonyms=False):
        """
        Retrieve all relationships connected to at least one of the specified concepts.
        Edges in the result graph present the following attributes:
            'rel' (type of relationship), 'weight' (strength of rel. according to Conceptnet)
        :param concepts: set of concepts
        :param antonyms: True if you want to include in the result Antonym relationships
        :return: Networkx graph with relationships that have one of the specified concepts in subject or reference
        """
        rows = self.df_conceptnet.loc[lambda x: (x[2].isin(concepts)) | (x[3].isin(concepts))]
        g = MultiDiGraph()
        for row in rows.iterrows():
            if row[1][1] != 'Antonym' or antonyms == True:
                jsondict = json.loads(row[1][4])  # Read edge information
                g.add_edge(row[1][2], row[1][3], rel=row[1][1], weight=float(jsondict['weight']))
        return g

    def get_related_places(self, concepts, antonyms=False):
        """
        Retrieve all relationships that connect one of the specified concept with a place.
        Specifically, selects from Conceptnets the triplets of the type:
        concept [RelatedTo, AtLocation, LocatedNear, PartOf, Antonym] place
        place [RelatedTo, AtLocation, LocatedNear, HasA, MadeOF, Antonym] concept
        Edges in the result graph present the following attributes:
            'rel' (type of relationship), 'weight' (strength of rel. according to Conceptnet)
        For "AtLocation", 'rel' contains also the position type (e.g. on, above, below, ...) retrieved from surface text
        :param concepts: set of concepts
        :param antonyms: True if you want to include Antonym relationships
        :return: Networkx graph with the selected relationships
        """
        conc_place_rel = {'RelatedTo', 'AtLocation', 'LocatedNear', 'PartOf'}
        place_conc_rel = {'RelatedTo', 'AtLocation', 'LocatedNear', 'HasA', 'MadeOf'}
        if antonyms:
            conc_place_rel.add('Antonym')
            place_conc_rel.add('Antonym')
        # Select relationships
        rows = self.df_conceptnet.loc[lambda x: (x[1].isin(conc_place_rel)
                                            & x[2].isin(concepts) & x[3].isin(self.places))
                                           | (x[1].isin(place_conc_rel)
                                              & x[2].isin(self.places) & x[3].isin(concepts))]
        g = MultiDiGraph()
        for row in rows.iterrows():
            t = ""
            jsondict = json.loads(row[1][4])  # Read edge information
            if row[1][1] == 'AtLocation' and 'surfaceText' in jsondict:
                s_text = jsondict['surfaceText']  # Surface text (sentence that describes the relationship)
                t = f" ({parse_loc_surface_text(s_text)[1]})"
            g.add_edge(row[1][2], row[1][3], rel=f"{row[1][1]}{t}", weight=float(jsondict['weight']))
        return g

    def __rank_related_places(self, concept_graph, concepts):
        """
        Use rank_related_places() instead of this function.
        Given a graph with conceptnet relationships (containing concepts and places),
        rank the places in the graph to find those that are most representative.

        The rank score encourages places that are more connected to other concepts in the graph.
             score(place)=sum( [max(edge_weights(place,concept)) for all concepts] )
        (max is necessary because there may be >1 edges between a place and a concept)

        :param concept_graph: Networkx graph with Conceptnet relationships
        :param concepts: list of concepts in the original COCO graph
        """
        related_places = {}
        # For each graph node that is a place
        for n in concept_graph.nodes:

            if n not in self.places:  # Must be a Conceptnet place
                continue
            # Get inbound and outbound edges of this node
            in_edges = [(e[0], e[2]['weight'], e[2]['rel']) for e in concept_graph.in_edges(n, data=True)]
            out_edges = [(e[1], e[2]['weight'], e[2]['rel']) for e in concept_graph.out_edges(n, data=True)]

            connections = {} # key=node id connected to n, value=list of weights
                             # (a pair place-concept may have multiple weights)

            found_antonym = False
            # Collect all edge weights for each connection
            for e in in_edges + out_edges:
                # Antonym: this place cannot be used
                if e[2] == 'Antonym':
                    found_antonym=True
                    break
                if e[0] in concepts:  # Connection place<->concept
                    if e[0] in connections:
                        connections[e[0]].append(-e[1] if e[2] == 'Antonym' else e[1])
                    else:
                        connections[e[0]] = [-e[1] if e[2] == 'Antonym' else e[1]]


            if not found_antonym:
                scores = [max(v) for v in connections.values()]
                # identity relationship for places that are also concepts
                if n in concepts:
                    scores.append(1)
                if len(scores)>1: # at least 2 edges
                    related_places[n] = sum(scores)

        return sorted(related_places.items(), key=lambda p: -p[1])

    def rank_related_places(self, json_graph, antonyms=True):
        """
        Given json_graph with COCO concepts,
        rank Conceptnet places to find those that are most representative.

        The rank score encourages places that are more connected to COCO concepts in the graph.
             score(place)=sum( [max(edge_weights(place,concept)) for all concepts] )
        (max is necessary because there may be >1 edges between a place and a concept)

        :param json_graph: json graph for which you want to rank places
        :param antonyms: True if you want to include Antonym relationships
                        (antonyms penalyze connection with places)
        """
        concepts = [coco_to_concept(n['label'], self.coco_concepts_map) for n in json_graph['nodes']]
        concept_graph = self.get_related_places(concepts, antonyms=antonyms)
        rank = self.__rank_related_places(concept_graph, concepts)
        return rank

def parse_loc_surface_text(text):
    """
    Parse SurfaceText (=Conceptnet text that generated a relationship)
    :param text: surface text
    :param subj: subject class
    :param ref: reference class
    :return: tuple with the position relationship i.e. ('vase', 'on', 'table')
    """
    m = re.search("\*Something you find (.+) \[\[(.+)\]\] is \[\[(.+)\]\]", text)
    if m is not None:
        return m.group(3).split(" ")[-1], m[1], m.group(2).split(" ")[-1]
    m = re.search("You are likely to find \[\[(.+)\]\] (.+) \[\[(.+)\]\]", text)
    if m is not None:
        return m.group(1).split(" ")[-1], m[2], m.group(3).split(" ")[-1]
    m = re.search("Somewhere \[\[(.+)\]\] can be is (.+) \[\[(.+)\]\]", text)
    if m is not None:
        return m.group(1).split(" ")[-1], m[2], m.group(3).split(" ")[-1]
    return None




