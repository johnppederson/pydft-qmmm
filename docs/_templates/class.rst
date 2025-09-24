{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   {% if methods|length > 1 %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
   {% if item != "__init__" %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block method_descriptions %}
   {% if methods %}
   {% for item in methods %}
   {% if item != "__init__" %}
   .. automethod:: {{ name }}.{{ item }}
   {% endif %}
   {% endfor %}
   {% endif %} 
   {% endblock %}
