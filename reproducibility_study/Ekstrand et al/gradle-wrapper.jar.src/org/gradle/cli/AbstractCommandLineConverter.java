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
/*    */ public abstract class AbstractCommandLineConverter<T>
/*    */   implements CommandLineConverter<T>
/*    */ {
/*    */   public T convert(Iterable<String> args, T target) throws CommandLineArgumentException {
/* 20 */     CommandLineParser parser = new CommandLineParser();
/* 21 */     configure(parser);
/* 22 */     return convert(parser.parse(args), target);
/*    */   }
/*    */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\cli\AbstractCommandLineConverter.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */