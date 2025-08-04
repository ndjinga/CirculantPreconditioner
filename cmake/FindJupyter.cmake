find_program(Jupyter_EXECUTABLE NAMES jupyter)

if(Jupyter_EXECUTABLE)
  message(STATUS "Found Jupyter: ${Jupyter_EXECUTABLE}")
else()
  message(WARNING "Jupyter executable not found. The 'notebook' target will not work until Jupyter is installed.")
endif()