/*    */ package org.gradle.cli;
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ public class ProjectPropertiesCommandLineConverter
/*    */   extends AbstractPropertiesCommandLineConverter
/*    */ {
/*    */   protected String getPropertyOption() {
/* 23 */     return "P";
/*    */   }
/*    */ 
/*    */   
/*    */   protected String getPropertyOptionDetailed() {
/* 28 */     return "project-prop";
/*    */   }
/*    */ 
/*    */   
/*    */   protected String getPropertyOptionDescription() {
/* 33 */     return "Set project property for the build script (e.g. -Pmyprop=myvalue).";
/*    */   }
/*    */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\cli\ProjectPropertiesCommandLineConverter.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */