from pxr import Usd, UsdGeom, Sdf, PhysxSchema

def export_geometry(
        points: list[tuple[float]], 
        vertex_counts: list[int],
        face_vertex_indices: list[int], 
        output_path: str,
        smooth: bool = False
        ) -> None:
     
    stage = Usd.Stage.CreateNew(f"./stages/{output_path}")
    root_xform = UsdGeom.Xform.Define(stage, "/Root")
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")

    # Create mesh attributes
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    if smooth:
        mesh.CreateSubdivisionSchemeAttr().Set("catmullClark")
    else:
        mesh.CreateSubdivisionSchemeAttr().Set("none")

    # Apply PhysX Collision API to the mesh
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    collision_api.CreateCollisionEnabledAttr(True)
    collision_api.CreateCollisionShapeAttr().Set("triangleMesh")

    # Save USD
    stage.GetRootLayer().Save()