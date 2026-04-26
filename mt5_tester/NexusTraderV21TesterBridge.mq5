#property strict
#property version   "1.00"
#property description "Replays exported Nexus Trader V21 signals inside MT5 Strategy Tester."

#include <Trade/Trade.mqh>

input string SignalCsvFile = "v21_mt5_tester_signals.csv";
input bool   UseCommonFiles = true;
input bool   RelaxedSymbolMatch = true;
input bool   CloseOnOppositeSignal = true;
input bool   AllowLong = true;
input bool   AllowShort = true;
input bool   OnePositionPerSymbol = true;
input ulong  MagicNumber = 21042026;
input double LotOverride = 0.0;
input int    SlippagePoints = 20;

struct TesterSignal
{
   int      id;
   datetime signal_time;
   datetime execution_time;
   string   action;
   double   lot;
   double   stop_loss;
   double   take_profit;
   string   confidence_tier;
   string   branch_label;
   bool     processed;
};

TesterSignal g_signals[];
int          g_signal_count = 0;
datetime     g_last_bar_time = 0;
CTrade       g_trade;

string NormalizeSymbolName(const string value)
{
   string upper = value;
   StringToUpper(upper);
   string normalized = "";
   for(int index = 0; index < StringLen(upper); index++)
   {
      ushort code = StringGetCharacter(upper, index);
      bool is_alpha = (code >= 'A' && code <= 'Z');
      bool is_digit = (code >= '0' && code <= '9');
      if(is_alpha || is_digit)
         normalized += CharToString(code);
   }
   return normalized;
}

bool SymbolMatches(const string csv_symbol, const string chart_symbol)
{
   string csv_upper = csv_symbol;
   string chart_upper = chart_symbol;
   StringToUpper(csv_upper);
   StringToUpper(chart_upper);
   if(csv_upper == chart_upper)
      return true;

   if(!RelaxedSymbolMatch)
      return false;

   string csv_normalized = NormalizeSymbolName(csv_symbol);
   string chart_normalized = NormalizeSymbolName(chart_symbol);
   if(csv_normalized == chart_normalized)
      return true;

   if(StringLen(csv_normalized) >= 6 && StringFind(chart_normalized, csv_normalized) >= 0)
      return true;
   if(StringLen(chart_normalized) >= 6 && StringFind(csv_normalized, chart_normalized) >= 0)
      return true;

   return false;
}

datetime ParseIsoTime(const string value)
{
   string trimmed = value;
   if(StringLen(trimmed) >= 19)
      trimmed = StringSubstr(trimmed, 0, 19);
   StringReplace(trimmed, "T", " ");
   return StringToTime(trimmed);
}

double NormalizeVolume(const double requested)
{
   double min_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double volume = requested;
   if(volume <= 0.0)
      volume = min_volume;
   if(step > 0.0)
      volume = MathFloor(volume / step) * step;
   volume = MathMax(volume, min_volume);
   volume = MathMin(volume, max_volume);
   return NormalizeDouble(volume, 2);
}

bool LoadSignals()
{
   int flags = FILE_READ | FILE_CSV | FILE_ANSI;
   if(UseCommonFiles)
      flags |= FILE_COMMON;

   int handle = FileOpen(SignalCsvFile, flags, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("V21 Tester Bridge: could not open signal CSV: ", SignalCsvFile, " error=", GetLastError());
      return false;
   }

   ArrayResize(g_signals, 0);
   g_signal_count = 0;
   int total_rows = 0;
   int matching_rows = 0;
   string seen_symbols = "";

   while(!FileIsEnding(handle))
   {
      string signal_id = FileReadString(handle);
      if(FileIsEnding(handle) && signal_id == "")
         break;

      string symbol_name = FileReadString(handle);
      string mode = FileReadString(handle);
      string signal_time = FileReadString(handle);
      string execution_time = FileReadString(handle);
      string action = FileReadString(handle);
      string lot = FileReadString(handle);
      string reference_close = FileReadString(handle);
      string execution_open = FileReadString(handle);
      string stop_loss = FileReadString(handle);
      string take_profit = FileReadString(handle);
      string stop_pips = FileReadString(handle);
      string take_profit_pips = FileReadString(handle);
      string confidence_tier = FileReadString(handle);
      string sqt_label = FileReadString(handle);
      string cabr_score = FileReadString(handle);
      string cpm_score = FileReadString(handle);
      string conformal_confidence = FileReadString(handle);
      string kelly_fraction = FileReadString(handle);
      string dangerous_branch_count = FileReadString(handle);
      string branch_label = FileReadString(handle);
      string execution_reason = FileReadString(handle);
      string final_summary = FileReadString(handle);

      if(StringLen(signal_id) > 0 && StringGetCharacter(signal_id, 0) == 65279)
         signal_id = StringSubstr(signal_id, 1);
      if(signal_id == "signal_id")
         continue;
      total_rows++;
      if(symbol_name != "")
      {
         string symbol_probe = "[" + symbol_name + "]";
         if(StringFind(seen_symbols, symbol_probe) < 0)
         {
            if(seen_symbols != "")
               seen_symbols += ", ";
            seen_symbols += symbol_probe;
         }
      }
      if(!SymbolMatches(symbol_name, _Symbol))
         continue;
      matching_rows++;

      TesterSignal item;
      item.id = (int)StringToInteger(signal_id);
      item.signal_time = ParseIsoTime(signal_time);
      item.execution_time = ParseIsoTime(execution_time);
      item.action = action;
      item.lot = StringToDouble(lot);
      item.stop_loss = StringToDouble(stop_loss);
      item.take_profit = StringToDouble(take_profit);
      item.confidence_tier = confidence_tier;
      item.branch_label = branch_label;
      item.processed = false;

      int next_index = ArraySize(g_signals);
      ArrayResize(g_signals, next_index + 1);
      g_signals[next_index] = item;
      g_signal_count++;
   }

   FileClose(handle);
   Print("V21 Tester Bridge: loaded ", g_signal_count, " signals for ", _Symbol, ". Total CSV rows=", total_rows, ", matched_rows=", matching_rows, ", seen_symbols=", seen_symbols, ", relaxed_match=", (RelaxedSymbolMatch ? "true" : "false"));
   return g_signal_count > 0;
}

bool HasOpenPosition()
{
   return PositionSelect(_Symbol);
}

ENUM_POSITION_TYPE CurrentPositionType()
{
   if(!PositionSelect(_Symbol))
      return WRONG_VALUE;
   return (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
}

bool CloseCurrentPosition()
{
   if(!PositionSelect(_Symbol))
      return true;
   return g_trade.PositionClose(_Symbol, SlippagePoints);
}

bool ExecuteSignal(TesterSignal &signal)
{
   string action = signal.action;
   StringToUpper(action);
   if((action == "BUY" && !AllowLong) || (action == "SELL" && !AllowShort))
      return true;

   bool has_position = HasOpenPosition();
   if(has_position)
   {
      ENUM_POSITION_TYPE current_type = CurrentPositionType();
      bool opposite = ((action == "BUY" && current_type == POSITION_TYPE_SELL) || (action == "SELL" && current_type == POSITION_TYPE_BUY));
      bool same_side = ((action == "BUY" && current_type == POSITION_TYPE_BUY) || (action == "SELL" && current_type == POSITION_TYPE_SELL));

      if(same_side && OnePositionPerSymbol)
      {
         Print("V21 Tester Bridge: skipping signal ", signal.id, " because a same-side position is already open.");
         return true;
      }
      if(opposite && CloseOnOppositeSignal)
      {
         if(!CloseCurrentPosition())
         {
            Print("V21 Tester Bridge: failed to close opposite position before signal ", signal.id, " error=", GetLastError());
            return false;
         }
      }
      else if(opposite && !CloseOnOppositeSignal)
      {
         Print("V21 Tester Bridge: skipping opposite signal ", signal.id, " because CloseOnOppositeSignal is disabled.");
         return true;
      }
   }

   double volume = NormalizeVolume(LotOverride > 0.0 ? LotOverride : signal.lot);
   double sl = signal.stop_loss > 0.0 ? NormalizeDouble(signal.stop_loss, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)) : 0.0;
   double tp = signal.take_profit > 0.0 ? NormalizeDouble(signal.take_profit, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)) : 0.0;

   g_trade.SetExpertMagicNumber((int)MagicNumber);
   g_trade.SetDeviationInPoints(SlippagePoints);
   g_trade.SetTypeFillingBySymbol(_Symbol);

   bool ok = false;
   if(action == "BUY")
      ok = g_trade.Buy(volume, _Symbol, 0.0, sl, tp, "Nexus V21 Tester");
   else if(action == "SELL")
      ok = g_trade.Sell(volume, _Symbol, 0.0, sl, tp, "Nexus V21 Tester");
   else
      return true;

   if(!ok)
   {
      Print("V21 Tester Bridge: order failed for signal ", signal.id, " action=", action, " lot=", DoubleToString(volume, 2), " error=", GetLastError());
      return false;
   }

   Print("V21 Tester Bridge: executed signal ", signal.id, " ", action, " lot=", DoubleToString(volume, 2), " branch=", signal.branch_label, " confidence=", signal.confidence_tier);
   return true;
}

void ProcessBar(const datetime bar_time)
{
   for(int i = 0; i < ArraySize(g_signals); i++)
   {
      if(g_signals[i].processed)
         continue;
      if(g_signals[i].execution_time != bar_time)
         continue;
      bool ok = ExecuteSignal(g_signals[i]);
      g_signals[i].processed = true;
      if(!ok)
         Print("V21 Tester Bridge: signal ", g_signals[i].id, " marked processed after failure to avoid duplicate orders.");
   }
}

int OnInit()
{
   if(!LoadSignals())
      return INIT_FAILED;
   return INIT_SUCCEEDED;
}

void OnTick()
{
   datetime current_bar = iTime(_Symbol, PERIOD_M15, 0);
   if(current_bar <= 0)
      return;
   if(current_bar == g_last_bar_time)
      return;
   g_last_bar_time = current_bar;
   ProcessBar(current_bar);
}
