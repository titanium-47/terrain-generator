from pxr import Usd, UsdGeom, Sdf

def export_geometry(
        points: list[tuple[float]], 
        vertex_counts: list[float],
        face_vertex_indices: list[float], 
        output_path: str,
        smooth: bool = False
        ) -> None:
     
    stage = Usd.Stage.CreateNew(f"./stages/{output_path}")
    root_xform = UsdGeom.Xform.Define(stage, "/Root")
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")

    #create mesh attributes
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    if smooth:
        mesh.CreateSubdivisionSchemeAttr().Set("catmullClark")
    else:
        mesh.CreateSubdivisionSchemeAttr().Set("none")

    #save USD
    stage.GetRootLayer().Save()
    
