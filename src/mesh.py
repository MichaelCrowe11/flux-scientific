"""
FLUX Mesh and Geometry Support
Mesh generation and management for scientific computing
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod

@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0
    
    def distance(self, other: 'Point') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

@dataclass
class Cell:
    """Mesh cell (element)"""
    id: int
    nodes: List[int]  # Node IDs
    type: str  # triangle, quad, tetrahedron, hexahedron
    material_id: int = 0

@dataclass
class Boundary:
    """Boundary definition"""
    name: str
    faces: List[Tuple[int, ...]]  # Face node IDs
    type: str = "wall"  # wall, inlet, outlet, symmetry, etc.

class Mesh(ABC):
    """Base mesh class"""
    def __init__(self, name: str):
        self.name = name
        self.nodes: List[Point] = []
        self.cells: List[Cell] = []
        self.boundaries: List[Boundary] = []
        
    @abstractmethod
    def generate(self) -> None:
        """Generate the mesh"""
        pass
    
    def get_node_count(self) -> int:
        return len(self.nodes)
    
    def get_cell_count(self) -> int:
        return len(self.cells)
    
    def add_node(self, x: float, y: float, z: float = 0.0) -> int:
        """Add a node and return its ID"""
        node_id = len(self.nodes)
        self.nodes.append(Point(x, y, z))
        return node_id
    
    def add_cell(self, nodes: List[int], cell_type: str, material_id: int = 0) -> int:
        """Add a cell and return its ID"""
        cell_id = len(self.cells)
        self.cells.append(Cell(cell_id, nodes, cell_type, material_id))
        return cell_id
    
    def add_boundary(self, name: str, faces: List[Tuple[int, ...]], boundary_type: str = "wall"):
        """Add a boundary condition"""
        self.boundaries.append(Boundary(name, faces, boundary_type))

class StructuredGrid(Mesh):
    """Structured Cartesian grid"""
    def __init__(self, name: str, x_range: Tuple[float, float], y_range: Tuple[float, float], 
                 nx: int, ny: int, z_range: Optional[Tuple[float, float]] = None, nz: int = 1):
        super().__init__(name)
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range or (0.0, 0.0)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
    def generate(self) -> None:
        """Generate structured grid"""
        # Create nodes
        x_coords = np.linspace(self.x_range[0], self.x_range[1], self.nx + 1)
        y_coords = np.linspace(self.y_range[0], self.y_range[1], self.ny + 1)
        
        if self.nz > 1:
            z_coords = np.linspace(self.z_range[0], self.z_range[1], self.nz + 1)
        else:
            z_coords = [0.0]
        
        # Add nodes
        node_map = {}
        for k, z in enumerate(z_coords):
            for j, y in enumerate(y_coords):
                for i, x in enumerate(x_coords):
                    node_id = self.add_node(x, y, z)
                    node_map[(i, j, k)] = node_id

        # Add cells
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    if self.nz > 1:  # 3D hexahedron
                        nodes = [
                            node_map[(i, j, k)], node_map[(i+1, j, k)],
                            node_map[(i+1, j+1, k)], node_map[(i, j+1, k)],
                            node_map[(i, j, k+1)], node_map[(i+1, j, k+1)],
                            node_map[(i+1, j+1, k+1)], node_map[(i, j+1, k+1)]
                        ]
                        self.add_cell(nodes, "hexahedron")
                    else:  # 2D quadrilateral
                        nodes = [
                            node_map[(i, j, k)], node_map[(i+1, j, k)],
                            node_map[(i+1, j+1, k)], node_map[(i, j+1, k)]
                        ]
                        self.add_cell(nodes, "quad")

        # Add boundaries
        self._create_boundaries(node_map)
    
    def _create_boundaries(self, node_map: Dict[Tuple[int, int, int], int]):
        """Create boundary faces"""
        # Left boundary (x = x_min)
        left_faces = []
        for k in range(self.nz):
            for j in range(self.ny):
                if self.nz == 1:
                    face = (node_map[(0, j, k)], node_map[(0, j+1, k)])
                else:
                    face = (node_map[(0, j, k)], node_map[(0, j+1, k)], 
                           node_map[(0, j+1, k+1)], node_map[(0, j, k+1)])
                left_faces.append(face)
        self.add_boundary("left", left_faces, "wall")
        
        # Right boundary (x = x_max)
        right_faces = []
        for k in range(self.nz):
            for j in range(self.ny):
                if self.nz == 1:
                    face = (node_map[(self.nx, j, k)], node_map[(self.nx, j+1, k)])
                else:
                    face = (node_map[(self.nx, j, k)], node_map[(self.nx, j+1, k)], 
                           node_map[(self.nx, j+1, k+1)], node_map[(self.nx, j, k+1)])
                right_faces.append(face)
        self.add_boundary("right", right_faces, "wall")
        
        # Bottom boundary (y = y_min)
        bottom_faces = []
        for k in range(self.nz):
            for i in range(self.nx):
                if self.nz == 1:
                    face = (node_map[(i, 0, k)], node_map[(i+1, 0, k)])
                else:
                    face = (node_map[(i, 0, k)], node_map[(i+1, 0, k)], 
                           node_map[(i+1, 0, k+1)], node_map[(i, 0, k+1)])
                bottom_faces.append(face)
        self.add_boundary("bottom", bottom_faces, "wall")
        
        # Top boundary (y = y_max)
        top_faces = []
        for k in range(self.nz):
            for i in range(self.nx):
                if self.nz == 1:
                    face = (node_map[(i, self.ny, k)], node_map[(i+1, self.ny, k)])
                else:
                    face = (node_map[(i, self.ny, k)], node_map[(i+1, self.ny, k)], 
                           node_map[(i+1, self.ny, k+1)], node_map[(i, self.ny, k+1)])
                top_faces.append(face)
        self.add_boundary("top", top_faces, "wall")

class UnstructuredMesh(Mesh):
    """Unstructured triangular/tetrahedral mesh"""
    def __init__(self, name: str, geometry: Any = None):
        super().__init__(name)
        self.geometry = geometry
        
    def generate(self) -> None:
        """Generate unstructured mesh - simplified implementation"""
        if self.geometry is None:
            # Create a simple triangular mesh for unit square
            self._generate_simple_triangular_mesh()
        else:
            # In a real implementation, this would use external meshing libraries
            # like GMSH, Tetgen, or Triangle
            raise NotImplementedError("Complex geometry meshing not implemented")
    
    def _generate_simple_triangular_mesh(self):
        """Generate simple triangular mesh for unit square"""
        # Create nodes for 3x3 grid
        for j in range(3):
            for i in range(3):
                x = i * 0.5
                y = j * 0.5
                self.add_node(x, y)
        
        # Create triangular cells
        for j in range(2):
            for i in range(2):
                # Node indices for current square
                n0 = j * 3 + i      # bottom-left
                n1 = j * 3 + i + 1  # bottom-right
                n2 = (j+1) * 3 + i  # top-left
                n3 = (j+1) * 3 + i + 1  # top-right
                
                # Two triangles per square
                self.add_cell([n0, n1, n2], "triangle")
                self.add_cell([n1, n3, n2], "triangle")
        
        # Add boundaries
        self._create_triangle_boundaries()
    
    def _create_triangle_boundaries(self):
        """Create boundaries for triangular mesh"""
        # Bottom boundary
        bottom_faces = [(0, 1), (1, 2)]
        self.add_boundary("bottom", bottom_faces, "wall")
        
        # Right boundary
        right_faces = [(2, 5), (5, 8)]
        self.add_boundary("right", right_faces, "wall")
        
        # Top boundary
        top_faces = [(6, 7), (7, 8)]
        self.add_boundary("top", top_faces, "wall")
        
        # Left boundary
        left_faces = [(0, 3), (3, 6)]
        self.add_boundary("left", left_faces, "wall")

class AdaptiveMesh(Mesh):
    """Adaptive mesh refinement"""
    def __init__(self, name: str, base_mesh: Mesh, refinement_criteria: Dict[str, Any]):
        super().__init__(name)
        self.base_mesh = base_mesh
        self.refinement_criteria = refinement_criteria
        self.refinement_levels = {}  # cell_id -> level
        
    def generate(self) -> None:
        """Generate adaptive mesh from base mesh"""
        # Start with base mesh
        self.nodes = self.base_mesh.nodes.copy()
        self.cells = self.base_mesh.cells.copy()
        self.boundaries = self.base_mesh.boundaries.copy()
        
        # Initialize refinement levels
        for cell in self.cells:
            self.refinement_levels[cell.id] = 0
    
    def refine_cell(self, cell_id: int) -> List[int]:
        """Refine a cell and return new cell IDs"""
        if cell_id >= len(self.cells):
            return []
        
        cell = self.cells[cell_id]
        current_level = self.refinement_levels[cell_id]
        max_level = self.refinement_criteria.get('max_level', 5)
        
        if current_level >= max_level:
            return []
        
        new_cells = []
        
        if cell.type == "triangle":
            new_cells = self._refine_triangle(cell)
        elif cell.type == "quad":
            new_cells = self._refine_quad(cell)
        
        # Update refinement levels
        for new_cell_id in new_cells:
            self.refinement_levels[new_cell_id] = current_level + 1
        
        # Mark original cell as inactive (in a real implementation)
        # For simplicity, we keep it active here
        
        return new_cells
    
    def _refine_triangle(self, cell: Cell) -> List[int]:
        """Refine triangular cell into 4 triangles"""
        # Get vertices
        n0, n1, n2 = cell.nodes
        p0, p1, p2 = self.nodes[n0], self.nodes[n1], self.nodes[n2]
        
        # Create midpoints
        mid01_id = self.add_node((p0.x + p1.x)/2, (p0.y + p1.y)/2, (p0.z + p1.z)/2)
        mid12_id = self.add_node((p1.x + p2.x)/2, (p1.y + p2.y)/2, (p1.z + p2.z)/2)
        mid02_id = self.add_node((p0.x + p2.x)/2, (p0.y + p2.y)/2, (p0.z + p2.z)/2)
        
        # Create 4 new triangles
        new_cells = []
        new_cells.append(self.add_cell([n0, mid01_id, mid02_id], "triangle"))
        new_cells.append(self.add_cell([n1, mid12_id, mid01_id], "triangle"))
        new_cells.append(self.add_cell([n2, mid02_id, mid12_id], "triangle"))
        new_cells.append(self.add_cell([mid01_id, mid12_id, mid02_id], "triangle"))
        
        return new_cells
    
    def _refine_quad(self, cell: Cell) -> List[int]:
        """Refine quadrilateral cell into 4 quads"""
        # Get vertices
        n0, n1, n2, n3 = cell.nodes
        p0, p1, p2, p3 = self.nodes[n0], self.nodes[n1], self.nodes[n2], self.nodes[n3]
        
        # Create edge midpoints
        mid01_id = self.add_node((p0.x + p1.x)/2, (p0.y + p1.y)/2, (p0.z + p1.z)/2)
        mid12_id = self.add_node((p1.x + p2.x)/2, (p1.y + p2.y)/2, (p1.z + p2.z)/2)
        mid23_id = self.add_node((p2.x + p3.x)/2, (p2.y + p3.y)/2, (p2.z + p3.z)/2)
        mid30_id = self.add_node((p3.x + p0.x)/2, (p3.y + p0.y)/2, (p3.z + p0.z)/2)
        
        # Create center point
        center_id = self.add_node(
            (p0.x + p1.x + p2.x + p3.x)/4,
            (p0.y + p1.y + p2.y + p3.y)/4,
            (p0.z + p1.z + p2.z + p3.z)/4
        )
        
        # Create 4 new quads
        new_cells = []
        new_cells.append(self.add_cell([n0, mid01_id, center_id, mid30_id], "quad"))
        new_cells.append(self.add_cell([mid01_id, n1, mid12_id, center_id], "quad"))
        new_cells.append(self.add_cell([center_id, mid12_id, n2, mid23_id], "quad"))
        new_cells.append(self.add_cell([mid30_id, center_id, mid23_id, n3], "quad"))
        
        return new_cells

class Field:
    """Field defined on a mesh"""
    def __init__(self, name: str, mesh: Mesh, location: str = "nodes"):
        self.name = name
        self.mesh = mesh
        self.location = location  # "nodes" or "cells"
        
        if location == "nodes":
            self.data = np.zeros(mesh.get_node_count())
        else:
            self.data = np.zeros(mesh.get_cell_count())
    
    def set_value(self, index: int, value: float):
        """Set field value at index"""
        self.data[index] = value
    
    def get_value(self, index: int) -> float:
        """Get field value at index"""
        return self.data[index]
    
    def set_boundary_condition(self, boundary_name: str, value: float):
        """Set boundary condition for the field"""
        for boundary in self.mesh.boundaries:
            if boundary.name == boundary_name:
                for face in boundary.faces:
                    for node_id in face:
                        if self.location == "nodes":
                            self.set_value(node_id, value)

def create_mesh(mesh_type: str, name: str, **kwargs) -> Mesh:
    """Factory function to create meshes"""
    if mesh_type == "StructuredGrid":
        domain = kwargs.get('domain', ((0, 1), (0, 1)))
        if isinstance(domain, tuple) and len(domain) == 2:
            x_range, y_range = domain
        else:
            x_range = (0, 1)
            y_range = (0, 1)
        
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        nz = kwargs.get('nz', 1)
        z_range = kwargs.get('z_range', None)
        
        mesh = StructuredGrid(name, x_range, y_range, nx, ny, z_range, nz)
        
    elif mesh_type == "UnstructuredMesh":
        geometry = kwargs.get('geometry', None)
        mesh = UnstructuredMesh(name, geometry)
        
    elif mesh_type == "AdaptiveMesh":
        base_mesh = kwargs.get('base_mesh')
        if not base_mesh:
            # Create default base mesh
            base_mesh = StructuredGrid("base", (0, 1), (0, 1), 10, 10)
            base_mesh.generate()
        
        refinement_criteria = kwargs.get('refinement_criteria', {'max_level': 3})
        mesh = AdaptiveMesh(name, base_mesh, refinement_criteria)
        
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")
    
    mesh.generate()
    return mesh