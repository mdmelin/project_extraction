from pathlib import Path
from ibllib.pipes.local_server import _get_pipeline_class
import projects.base
# how to run a single task
from one.api import ONE

one = ONE()

# how to run a pipeline
session_path = Path("/mnt/s0/Data/Subjects/KS056/2021-07-18/001")
session_path = Path("/media/olivier/Extreme SSD/KS056/2021-07-18/001")
PipelineClass = projects.base.get_pipeline(session_path)
pipe = PipelineClass(session_path, one=one)
pipe.make_graph()


from projects.ephys_passive_opto import EphysPassiveOptoPipeline
session_path = Path("/media/olivier/Extreme SSD/KS056/2021-07-18/001")

# how to run a single task
for tname in pipe.tasks:
    if 'RegisterRaw' in tname:
        continue
    print(tname)
    break