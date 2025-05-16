from typing import Set
from fiction.yagottl.TurtleUtils import Graph


def class_superclasses(cls: str, taxonomy: Graph) -> Set[str]:
    """Get all the superclasses of CLS"""
    parents = taxonomy.index.get(cls, {}).get("rdfs:subClassOf", set())
    superclasses = set(parents)
    for parent in parents:
        superclasses |= class_superclasses(parent, taxonomy)
    return superclasses


def entity_types(entity: str, facts: Graph) -> Set[str]:
    """Return the direct types of ENTITY

    .. note::

        NOTE: this is a safe way of getting the type, that does not crash
        if the type is unknown because a fact is missing (in that case, it
        will return the empty set)
    """
    return facts.index.get(entity, {}).get("rdf:type", set())


def allowed_rels(subj: str, facts: Graph, schema: Graph, taxonomy: Graph) -> Set[str]:
    """Return the set of allowed relations for SUBJ"""
    allowed_relations = set()
    types = set()
    for typ in entity_types(subj, facts):
        types |= class_superclasses(typ, taxonomy)
    for typ in types:
        for prop in schema.index.get(typ, {}).get("sh:property", set()):
            for prop_path in schema.index[prop]["sh:path"]:
                allowed_relations.add(prop_path)
    return allowed_relations


def is_rel_allowed(
    subj: str, rel: str, facts: Graph, schema: Graph, taxonomy: Graph
) -> bool:
    """Check whether SUBJ is allowed to have relation REL"""
    return rel in allowed_rels(subj, facts, schema, taxonomy)


def allowed_obj_types(rel: str, schema: Graph) -> Set[str]:
    """Return the set of object types allowed for relation REL"""
    # get all properties corresponding to rel
    assert not schema.inverseGraph is None
    properties = schema.inverseGraph.index[rel].get("sh:path", set())
    # for each property, get the allowed class
    allowed_classes = set()
    for prop in properties:
        allowed_classes |= schema.index[prop].get("sh:class", set())
    return allowed_classes


def is_subclass(cls: str, superclass: str, taxonomy: Graph) -> bool:
    """Check whether CLS is a subclass of SUPERCLASS"""
    return superclass in class_superclasses(cls, taxonomy)


def is_obj_allowed(
    obj: str, rel: str, facts: Graph, schema: Graph, taxonomy: Graph
) -> bool:
    """Check whether OBJ is allowed for relation REL"""
    allowed_types = allowed_obj_types(rel, schema)
    for obj_type in entity_types(obj, facts):
        if any(
            is_subclass(obj_type, allowed_type, taxonomy)
            for allowed_type in allowed_types
        ):
            return True
    return False
