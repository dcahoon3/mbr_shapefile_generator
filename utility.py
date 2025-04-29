import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import orient


REQUIRED_HEADERS = ['customerid', 'zoneid', 'suffixid', 'areanumber', 'seqno', 'x', 'y']

def zoneid_suffixid_combine(row: pd.Series) -> str:
    """Combines zoneid and suffixid columns in a dataframe into a single string.
    The suffixid is appended to the zoneid with an underscore if it is not null or empty.

    Args:
        row (pd.Series): A row from a pandas dataframe containing zoneid and suffixid columns

    Returns:
        str: Combined zoneid_suffixid string. If suffixid is 'NONE', '', 'NULL', 'NAN', or None, only zoneid is returned.
    """
    suffix = str(row['suffixid']).strip().upper()
    if suffix in ['NONE', '', 'NULL', 'NAN', None]:
        return str(row['zoneid'])
    return f"{row['zoneid']}_{row['suffixid']}"

def split_geometry(df: pd.DataFrame) -> dict:
    """Parses the geometry for a areanumber group. The first section of coordinates
    is considered the exterior of the polygon, and any subsequent sections are considered holes.
    The function assumes that the coordinates are ordered by seqno and only represent a single areanumber

    Args:
        df (pandas.DataFrame): dataframe containing the geometry data for a single areanumber

    Returns:
        dict: dict with exterior and holes as keys. Each key contains a list of tuples representing the coordinates.
        If no valid geometry is found, returns None.
    """
    coords = df[['x', 'y']].values.tolist()
    parts = []
    current = []

    for x, y in coords:
        if (x, y) == (0, 0):
            if current:
                parts.append(current)
                current = []
        else:
            current.append((x, y))
    if current:
        parts.append(current)

    if not parts:
        return None

    return {
        'exterior': parts[0],
        'holes': parts[1:] if len(parts) > 1 else []
    }


def build_multipolygon(group_df: pd.DataFrame) -> Polygon or MultiPolygon: # type: ignore
    """Creates polygon geometry from a dataframe. This function assumes that the dataframe only contains a single
    zoneid_suffixid. The function first groups the dataframe by areanumber and then sorts the coordinates by seqno.
    It then calls the split_geometry function to parse the coordinates into exterior and hole parts.
    Finally, it creates a Polygon or MultiPolygon object from the parsed coordinates.
    If no valid geometry is found, returns None.

    Args:
        group_df (pd.DataFrame): dataframe containing the geometry data for a single zoneid_suffixid

    Returns:
        Polygon or MultiPolygon: Polygon or MultiPolygon object representing the geometry.
        If no valid geometry is found, returns None.
    """
    polygons = []

    for _, area_group in group_df.groupby('areanumber'):
        area_group = area_group.sort_values('seqno')
        geom_parts = split_geometry(area_group)

        if geom_parts:
            poly = Polygon(geom_parts['exterior'], holes=geom_parts['holes'])
            poly = orient(poly, sign=1.0)  # Fix winding
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)

    if not polygons:
        return None
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)
