:github_url: {{ fullname }}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: .
   {% for item in attributes %}
     {%- if item not in inherited_members %}
       ~{{ fullname }}.{{ item }}
     {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: .
   {% for item in methods %}
      {%- if item != '__init__' %}
        {%- if item not in inherited_members %}
          ~{{ fullname }}.{{ item }}
        {%- endif -%}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
