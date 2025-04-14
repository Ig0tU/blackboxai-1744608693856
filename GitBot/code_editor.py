# code_editor.py

import json

def code_editor(value: str = "", issue_num: int = 0) -> str:
    """Returns an HTML snippet for a CodeMirror-based code editor."""
    """
    Returns an HTML snippet for a CodeMirror-based code editor.

    Args:
        value (str): Initial content for the code editor.
        issue_num (int): The issue number to identify the editor instance.

    Returns:
        str: HTML string that embeds a CodeMirror editor.
    """
    html = f"""  # HTML snippet for the CodeMirror editor
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.3/codemirror.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.3/theme/material-darker.min.css">
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.3/codemirror.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.3/mode/python/python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.3/mode/javascript/javascript.min.js"></script>
    
    <style>
      /* Basic styling for the editor container */
      #code-editor-container-{issue_num} {{
          border: 1px solid #ccc;
          border-radius: 4px;
          overflow: hidden;
          margin: 0 auto;
      }}
      .CodeMirror {{
          height: auto;
          min-height: 300px;
      }}
    </style>
    
    <div id="code-editor-container-{issue_num}">
        <textarea id="code-editor-{issue_num}">{value}</textarea>
    </div>
    
    <script>
      // Initialize CodeMirror on the textarea after it is loaded
      document.addEventListener("DOMContentLoaded", function() {{
          var editor = CodeMirror.fromTextArea(document.getElementById("code-editor-{issue_num}"), {{
              lineNumbers: true,
              mode: "python",  // Change this to "javascript" or any other mode as needed
              theme: "material-darker",
              tabSize: 4
          }});
          
          // Expose the editor globally so we can interact with it elsewhere if needed
          window._codeEditorInstance{issue_num} = editor;

          // Handle changes and send updates to the server
          editor.on("change", function() {{
              const delta = editor.getValue();
              const message = JSON.stringify({{
                  type: "code_update",
                  issue_num: {issue_num},
                  delta: delta
              }});
              // Send the updated code to the WebSocket server
              collabWs.send(message);
          }});
      }});
    </script>
    """
    return html
