package edu.boisestate.coen.piret.democool

import org.gradle.api.DefaultTask
import org.gradle.api.tasks.InputFile
import org.gradle.api.tasks.TaskAction

class NotebookExec extends DefaultTask {
    def notebook

    @InputFile
    File getNotebookFile() {
        return notebook ? project.file(notebook) : null
    }

    @TaskAction
    void runNotebook() {
        logger.info('exporting {} to script', notebookFile)
        // TODO Don't assume it is an R script
        def scriptName = notebookFile.name.replace('.ipynb', '.r')
        project.exec {
            executable 'jupyter'
            args 'nbconvert'
            args '--to', 'script'
            args '--output-dir', project.buildDir
            args notebookFile
        }
        logger.info('running script {}', scriptName)
        project.exec {
            executable 'Rscript'
            args "$project.buildDir/$scriptName"
        }
    }
}
