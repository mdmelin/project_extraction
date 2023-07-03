from iblrig_custom_tasks.plau_oddBallAudio.task import Session
task = Session(subject='toto')
task.start_hardware()

task._run()