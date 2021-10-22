package org.gradle.cli;

public interface CommandLineConverter<T> {
  T convert(Iterable<String> paramIterable, T paramT) throws CommandLineArgumentException;
  
  T convert(ParsedCommandLine paramParsedCommandLine, T paramT) throws CommandLineArgumentException;
  
  void configure(CommandLineParser paramCommandLineParser);
}


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\cli\CommandLineConverter.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */