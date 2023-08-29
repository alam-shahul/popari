from typing import Union, Sequence, Optional, Tuple
from collections import defaultdict
import json

import numpy as np
from scipy.spatial import KDTree

from ipycanvas import Canvas, hold_canvas
from ipywidgets import Output


class DomainKDTree:
    """Wrapper of KDTree for spatial domain simulation.
    
    Using these, can query a "landmark" KD tree in order to obtain spatial domain labels for simulation.
    """
    
    def __init__(self, landmarks: np.ndarray, domain_labels: Sequence[Union[int, str]]):
        self.kd_tree = KDTree(data=landmarks, copy_data=True)
        self.domain_labels = np.array(domain_labels)
        
    def query(self, coordinates: np.ndarray):
        """Query a list of simulation spatial coordinates for their layer label.
       
        Args:
            coordinates: list of spatial coordinates
        
        Returns:
            list of domain labels
        """
        distances, indices = self.kd_tree.query(coordinates)
        
        return self.domain_labels[indices]

class DomainCanvas():
    """Annotate spatial domains on a reference ST dataset and use to define domains for simulations.
    
    """
    
    def __init__(self, points: np.ndarray, domain_names: Sequence[str], canvas_width: int = 400,
            density: float = 10, precision=5):
        """Create DomainCanvas.
        
        Args:
            points (nd.array): background points to use as reference for domain annotation
            domain_names (list of str): list of domains that will be annotated
            canvas_width (int): display width of canvas in Jupyter Notebook, in pixels
            density (float): Try 1 for Visium and 4 for Slide-seq V2
        """
        self.precision = precision
        self.points = np.around(points, decimals=self.precision)
        self.set_domain_names(domain_names)

        self.domain_deletion_string = \
            "This domain has already been annotated. What would you like to do? (enter 0 or 1) \n" \
            "0: Further annotate it\n" \
            "1: Delete the existing annotation and start from scratch\n"       
        
        dimensions = self.points.ptp(axis=0)
        lower_boundaries = self.points.min(axis=0)
        self.width, self.height = dimensions + lower_boundaries
 
        self.canvas_width = canvas_width
        self.canvas_height = int((self.height / self.width) * self.canvas_width)

        self.scaling_factor = self.canvas_width / self.width
        
        self.canvas = Canvas(width=self.width * self.scaling_factor, height=self.height * self.scaling_factor)

        self.canvas.layout.width = f"{self.canvas_width}px"
        self.canvas.layout.height = f"{self.canvas_height}px"
        
        self.density = density
        self.radius = 6 / self.density       
        self.canvas.line_width = 2.5 / self.density
        
        self.render_reference_points()
        
        self.domains = defaultdict(list)
        self.colors = {}
        self.out = self.bind_canvas()
        
        return
   
    def set_domain_names(self, domain_names: Sequence):
        self.domain_names = domain_names
        self.domain_option_string = "\n".join(f"{index}: {domain}" for index, domain in enumerate(self.domain_names))

    def render_reference_points(self):
        x, y = self.points.T * self.scaling_factor

        self.canvas.stroke_style = "gray"
        self.draw_points(x, y)
        
    def redraw(self):
        self.canvas.clear()
        self.render_reference_points()
        self.load_domains(self.domains)
    
    def draw_point(self, x, y):
        with hold_canvas():
            self.canvas.sync_image_data=True
            self.canvas.stroke_circle(x, y, self.radius)
            self.canvas.fill_circle(x, y, self.radius)
            self.canvas.stroke()
            self.canvas.fill()
            self.canvas.sync_image_data=False
            
    def draw_points(self, x, y, fill=False):
        self.canvas.stroke_circles(x, y, self.radius)
        if fill:
            self.canvas.fill_circles(x, y, self.radius)
                
    def bind_canvas(self):
        """Bind mouse click to canvas.
        
        Only used during initialization.
        """
        out = Output()

        @out.capture()
        def handle_mouse_down(x, y):
            self.draw_point(x, y)
                
            self.domains[self.current_domain].append(np.around((x / self.scaling_factor, y / self.scaling_factor), decimals=self.precision))
            return

        self.canvas.on_mouse_down(handle_mouse_down)
        
        return out

    def display(self):
        """Display editable canvas.
        
        Click to add landmarks for the domain self.current_domain.
        
        """
        display(self.out)
        return self.canvas
    
    def annotate_domain(self, points: Sequence[Tuple[float, float]] = None):
        """Create a new domain and display the canvas for annotation.
        
        """
        domain_index = int(input(
            "Choose a domain to annotate (enter an integer):\n"
            f"{self.domain_option_string}\n"
        ))
        if not (0 <= domain_index < len(self.domain_names)):
            raise ValueError(f"`{domain_index}` is not a valid index.")
                                              
        self.current_domain = self.domain_names[domain_index]
        
        if self.current_domain in self.colors:
            start_afresh = int(input(self.domain_deletion_string))
            if start_afresh not in (0, 1):
                raise ValueError(f"`{start_afresh}` is not a valid option.")
                
            if start_afresh:
                del self.domains[self.current_domain]
                self.redraw()
                
            color = self.colors[self.current_domain]
        else:
            r, g, b = np.random.randint(0, 255, size=3)
            color = f"rgb({r}, {g}, {b})"
            self.colors[self.current_domain] = color
            
        self.canvas.stroke_style = color
        self.canvas.fill_style = color
        
        if points is not None:
            print("Appending predefined `points` to domain...")
            x, y = np.around(points, decimals=self.precision)
            self.draw_points(x * self.scaling_factor, y * self.scaling_factor, fill=True)
            
            self.domains[self.current_domain].extend(list(zip(x, y)))
        
        return self.display()
    
    def load_domains(self, domains):
        """Load and display a pre-defined set of domains.
        
        """

        for domain_name in domains:
            self.domains[domain_name] = domains[domain_name]
            coordinates = domains[domain_name]
            
            if domain_name in self.colors:
                color = self.colors[domain_name]
            else:
                r, g, b = np.random.randint(0, 255, size=3)
                color = f"rgb({r}, {g}, {b})"
                self.colors[domain_name] = color

            self.canvas.stroke_style = color
            self.canvas.fill_style = color
            
            x, y = np.array(coordinates).T * self.scaling_factor
            self.draw_points(x, y, fill=True)
    
    def generate_domain_kd_tree(self):
        """Export annotated dataset to KD-tree.
        
        """
        domains, coordinates = zip(*self.domains.items())

        domain_labels = [[domain] * len(coordinate) for domain, coordinate in zip(domains, coordinates)]

        flattened_domain_labels = np.concatenate(domain_labels)
        flattened_coordinates = np.concatenate(coordinates)
        annotated_domain_kd_tree = DomainKDTree(flattened_coordinates, flattened_domain_labels)
        
        return annotated_domain_kd_tree

class MetageneCanvas(DomainCanvas):
    def convert_metagenes_to_cell_types(self):
        """Assuming that the previous domain annotations are metagenes,
        converts these to unique cell types based on presence of coannotated metagenes.
        """
        cell_type_mapping = defaultdict(list)
        for domain, points in self.domains.items():
            for point in points:
                cell_type_mapping[point].append(domain)
        
        for point in self.points:
            point = tuple(point)
            if point not in cell_type_mapping:
                cell_type_mapping[point].append(None)
                
        reverse_mapping = defaultdict(list)
        for point, domain_list in cell_type_mapping.items():
            joint_domain = json.dumps(tuple(domain_list))
            reverse_mapping[joint_domain].append(point)
        
        self.domains = reverse_mapping
        self.redraw()

