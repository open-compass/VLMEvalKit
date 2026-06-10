"""
Options class for LaTeX parsing - Python version of KaTeX Options.js
This file contains information about the options that the Parser carries
around with it while parsing.
"""


class Options:
    """
    Main options class. It contains the style, size, color, and font
    of the current parse level. It also contains the style and size of the parent
    parse level, so size changes can be handled efficiently.
    """
    
    def __init__(self, style=None, size=None, color=None, phantom=None, font=None, 
                 parent_style=None, parent_size=None):
        self.style = style
        self.color = color
        self.size = size
        self.phantom = phantom
        self.font = font
        
        if parent_style is None:
            self.parent_style = style
        else:
            self.parent_style = parent_style
            
        if parent_size is None:
            self.parent_size = size
        else:
            self.parent_size = parent_size
    
    def extend(self, **extension):
        """
        Returns a new options object with the same properties as "this".
        Properties from "extension" will be copied to the new options object.
        """
        data = {
            'style': self.style,
            'size': self.size,
            'color': self.color,
            'parent_style': self.style,
            'parent_size': self.size,
            'phantom': self.phantom,
            'font': self.font,
        }
        data.update(extension)
        return Options(**data)
    
    def with_style(self, style):
        """Create a new options object with the given style."""
        return self.extend(style=style)
    
    def with_size(self, size):
        """Create a new options object with the given size."""
        return self.extend(size=size)
    
    def with_color(self, color):
        """Create a new options object with the given color."""
        return self.extend(color=color)
    
    def with_phantom(self):
        """Create a new options object with "phantom" set to true."""
        return self.extend(phantom=True)
    
    def with_font(self, font):
        """Create a new options objects with the give font."""
        return self.extend(font=font)
    
    def reset(self):
        """
        Create a new options object with the same style, size, and color.
        This is used so that parent style and size changes are handled correctly.
        """
        return self.extend()
    
    def get_color(self):
        """
        Gets the CSS color of the current options object, accounting for the
        colorMap.
        """
        if self.phantom:
            return "transparent"
        else:
            # Color map (simplified version, full map available in original Options.js)
            color_map = {
                "katex-blue": "#6495ed",
                "katex-orange": "#ffa500",
                "katex-pink": "#ff00af",
                "katex-red": "#df0030",
                "katex-green": "#28ae7b",
                "katex-gray": "gray",
                "katex-purple": "#9d38bd",
            }
            return color_map.get(self.color, self.color)

