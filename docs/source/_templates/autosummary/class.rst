:github_url: {{ fullname }}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: .
   {% for item in methods %}
      {%- if item != '__init__' %}
        {%- if item not in inherited_members %}
          ~{{ objname }}.{{ item }}
        {%- endif -%}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
